/*
 * ═══════════════════════════════════════════════════════════
 *  SMART PLAYGROUND MONITOR — ESP32-CAM FINAL VERSION
 * ═══════════════════════════════════════════════════════════
 * 
 * Features:
 * ✅ Maximum stable streaming for AI detection
 * ✅ VGA 640x480 @ 10-12 quality (best for ML)
 * ✅ WiFi sleep disabled (no latency spikes)
 * ✅ PSRAM optimized
 * ✅ Auto-reconnect WiFi
 * ✅ Quality control via HTTP (no reflash needed)
 * ✅ Status endpoint for monitoring
 * 
 * Endpoints:
 *   http://[IP]/stream   — MJPEG live stream
 *   http://[IP]/capture  — Single JPEG snapshot
 *   http://[IP]/status   — JSON status
 *   http://[IP]/quality  — Change quality/brightness/flip
 * 
 * ═══════════════════════════════════════════════════════════
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "fb_gfx.h"
#include "esp_http_server.h"

// ═══════════════════════════════════════════════════════════
// 📡 WiFi Configuration — CHANGE THESE
// ═══════════════════════════════════════════════════════════
const char* ssid     = "Moon";
const char* password = "Moonoon@07";
// ═══════════════════════════════════════════════════════════
// 📷 Camera Pins — AI-Thinker ESP32-CAM
// ═══════════════════════════════════════════════════════════
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define LED_GPIO_NUM       4

// ═══════════════════════════════════════════════════════════
// 🎥 Stream Configuration
// ═══════════════════════════════════════════════════════════
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY =
    "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ═══════════════════════════════════════════════════════════
// 📊 Global Variables
// ═══════════════════════════════════════════════════════════
httpd_handle_t stream_httpd = NULL;
unsigned long  frameCount   = 0;
static int     jpeg_quality = 10;  // Default: stable AI quality

// ═══════════════════════════════════════════════════════════
// 🔧 CAMERA INIT — AI-Optimized Settings
// ═══════════════════════════════════════════════════════════

bool initCamera() {
    camera_config_t config;

    // Pin configuration
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;

    // Clock and format
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    // ⚡ CRITICAL: Grab latest frame (not wait for old ones)
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;

    // ✅ AI-Optimized: VGA + quality 10 = best balance
    if (psramFound()) {
        config.frame_size   = FRAMESIZE_VGA;   // 640x480
        config.jpeg_quality = 10;              // Stable, clear, fast
        config.fb_count     = 2;               // Double buffer
        Serial.println("✅ PSRAM: VGA 640x480 quality=10 (AI OPTIMAL)");
    } else {
        config.frame_size   = FRAMESIZE_CIF;   // 400x296
        config.jpeg_quality = 12;              // Lighter for no PSRAM
        config.fb_count     = 1;
        Serial.println("⚠️ No PSRAM: CIF 400x296 quality=12");
    }

    // Initialize camera
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("❌ Camera init failed: 0x%x\n", err);
        return false;
    }

    // ═══════════════════════════════════════════════════
    // 🎛️ SENSOR SETTINGS — Natural for AI
    // ═══════════════════════════════════════════════════
    sensor_t* s = esp_camera_sensor_get();
    if (!s) {
        Serial.println("❌ Sensor not found");
        return false;
    }

    // Orientation — adjust if camera is mounted upside down
    s->set_vflip(s, 0);   // 0=normal, 1=flip vertical
    s->set_hmirror(s, 0); // 0=normal, 1=mirror

    // ✅ Natural settings (not over-processed)
    s->set_brightness(s, 0);   // -2 to 2, 0 = natural
    s->set_contrast(s, 0);     // -2 to 2, 0 = natural
    s->set_saturation(s, 0);   // -2 to 2, 0 = natural
    s->set_sharpness(s, 1);    // 0-2, 1 = moderate (not max)

    s->set_special_effect(s, 0); // 0 = no effect

    // White balance — AUTO (best for changing light)
    s->set_whitebal(s, 1);     // 1 = auto WB
    s->set_awb_gain(s, 1);     // Enable AWB gain
    s->set_wb_mode(s, 0);      // 0 = auto mode

    // Exposure — AUTO but stable (not aggressive)
    s->set_exposure_ctrl(s, 1);  // 1 = auto exposure
    s->set_aec2(s, 0);           // 0 = standard auto (not aggressive)
    s->set_ae_level(s, 0);       // 0 = normal exposure level

    // Gain — AUTO but limited (prevents noise)
    s->set_gain_ctrl(s, 1);           // 1 = auto gain
    s->set_agc_gain(s, 0);            // Initial gain
    s->set_gainceiling(s, (gainceiling_t)4); // Max gain = 4x (not 6x)

    // Noise reduction
    s->set_bpc(s, 1);   // Black pixel correction
    s->set_wpc(s, 1);   // White pixel correction
    s->set_raw_gma(s, 1); // Raw gamma
    s->set_lenc(s, 1);   // Lens correction

    s->set_dcw(s, 0);      // No downsize
    s->set_colorbar(s, 0); // MUST be 0 for real image

    Serial.println("✅ Camera initialized — AI-optimized settings");
    return true;
}

// ═══════════════════════════════════════════════════════════
// 📡 STREAM HANDLER — MJPEG Live Feed
// ═══════════════════════════════════════════════════════════

static esp_err_t stream_handler(httpd_req_t* req) {
    camera_fb_t* fb = NULL;
    esp_err_t res = ESP_OK;
    size_t jpg_len = 0;
    uint8_t* jpg_buf = NULL;
    char part_buf[64];

    // Set headers
    res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");

    Serial.println("📡 Stream client connected");

    while (true) {
        // Grab latest frame (not wait)
        fb = esp_camera_fb_get();

        if (!fb) {
            res = ESP_FAIL;
            Serial.println("❌ Frame grab failed");
            break;
        }

        // Convert if not JPEG
        if (fb->format != PIXFORMAT_JPEG) {
            bool ok = frame2jpg(fb, jpeg_quality, &jpg_buf, &jpg_len);
            esp_camera_fb_return(fb);
            fb = NULL;
            if (!ok) {
                res = ESP_FAIL;
                break;
            }
        } else {
            jpg_len = fb->len;
            jpg_buf = fb->buf;
        }

        // Send boundary
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
        }

        // Send header
        if (res == ESP_OK) {
            size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, jpg_len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        }

        // Send image data
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char*)jpg_buf, jpg_len);
        }

        // Cleanup
        if (fb) {
            esp_camera_fb_return(fb);
            fb = NULL;
        } else if (jpg_buf) {
            free(jpg_buf);
            jpg_buf = NULL;
        }

        if (res != ESP_OK) break;

        frameCount++;

        // 🛡️ Prevent buffer overflow — drop old frames
        if (frameCount > 1000000) {
            frameCount = 0;
        }
    }

    Serial.println("📡 Stream client disconnected");
    return res;
}

// ═══════════════════════════════════════════════════════════
// 📸 CAPTURE HANDLER — Single Snapshot
// ═══════════════════════════════════════════════════════════

static esp_err_t capture_handler(httpd_req_t* req) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    esp_err_t res = httpd_resp_send(req, (const char*)fb->buf, fb->len);

    esp_camera_fb_return(fb);
    return res;
}

// ═══════════════════════════════════════════════════════════
// 📊 STATUS HANDLER — JSON Status
// ═══════════════════════════════════════════════════════════

static esp_err_t status_handler(httpd_req_t* req) {
    char json[512];
    sensor_t* s = esp_camera_sensor_get();

    snprintf(json, sizeof(json),
        "{"
        "\"status\":\"online\","
        "\"ip\":\"%s\","
        "\"rssi\":%d,"
        "\"heap_free\":%d,"
        "\"frame_count\":%lu,"
        "\"uptime_sec\":%lu,"
        "\"psram\":%s,"
        "\"frame_size\":\"%s\","
        "\"jpeg_quality\":%d,"
        "\"fb_count\":%d"
        "}",
        WiFi.localIP().toString().c_str(),
        WiFi.RSSI(),
        (int)ESP.getFreeHeap(),
        frameCount,
        millis() / 1000,
        psramFound() ? "true" : "false",
        psramFound() ? "VGA" : "CIF",
        jpeg_quality,
        psramFound() ? 2 : 1
    );

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json, strlen(json));
}

// ═══════════════════════════════════════════════════════════
// 🎛️ QUALITY HANDLER — Change Settings Without Reflash
// ═══════════════════════════════════════════════════════════
// Usage:
//   http://[IP]/quality?val=10       (JPEG quality 2-63)
//   http://[IP]/quality?bright=0     (brightness -2 to 2)
//   http://[IP]/quality?contrast=0   (contrast -2 to 2)
//   http://[IP]/quality?sharp=1      (sharpness 0-2)
//   http://[IP]/quality?flip=1       (vertical flip 0/1)
//   http://[IP]/quality?mirror=1     (mirror 0/1)
// ═══════════════════════════════════════════════════════════

static esp_err_t quality_handler(httpd_req_t* req) {
    char query[128];
    char val[16];

    httpd_req_get_url_query_str(req, query, sizeof(query));

    sensor_t* s = esp_camera_sensor_get();
    if (!s) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    // JPEG quality
    if (httpd_query_key_value(query, "val", val, sizeof(val)) == ESP_OK) {
        int q = constrain(atoi(val), 2, 63);
        jpeg_quality = q;
        s->set_quality(s, q);
        Serial.printf("✅ JPEG quality: %d\n", q);
    }

    // Brightness
    if (httpd_query_key_value(query, "bright", val, sizeof(val)) == ESP_OK) {
        int b = constrain(atoi(val), -2, 2);
        s->set_brightness(s, b);
        Serial.printf("✅ Brightness: %d\n", b);
    }

    // Contrast
    if (httpd_query_key_value(query, "contrast", val, sizeof(val)) == ESP_OK) {
        int c = constrain(atoi(val), -2, 2);
        s->set_contrast(s, c);
        Serial.printf("✅ Contrast: %d\n", c);
    }

    // Sharpness
    if (httpd_query_key_value(query, "sharp", val, sizeof(val)) == ESP_OK) {
        int sh = constrain(atoi(val), 0, 2);
        s->set_sharpness(s, sh);
        Serial.printf("✅ Sharpness: %d\n", sh);
    }

    // Vertical flip
    if (httpd_query_key_value(query, "flip", val, sizeof(val)) == ESP_OK) {
        s->set_vflip(s, atoi(val));
        Serial.printf("✅ VFlip: %d\n", atoi(val));
    }

    // Mirror
    if (httpd_query_key_value(query, "mirror", val, sizeof(val)) == ESP_OK) {
        s->set_hmirror(s, atoi(val));
        Serial.printf("✅ Mirror: %d\n", atoi(val));
    }

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
    return ESP_OK;
}

// ═══════════════════════════════════════════════════════════
// 🌐 START HTTP SERVER
// ═══════════════════════════════════════════════════════════

void startServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.max_open_sockets = 5;  // Allow multiple clients
    config.stack_size = 8192;

    httpd_uri_t uris[] = {
        {"/stream",  HTTP_GET, stream_handler,  NULL},
        {"/capture", HTTP_GET, capture_handler, NULL},
        {"/status",  HTTP_GET, status_handler,  NULL},
        {"/quality", HTTP_GET, quality_handler, NULL}
    };

    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        for (int i = 0; i < 4; i++) {
            httpd_register_uri_handler(stream_httpd, &uris[i]);
        }
        Serial.println("✅ HTTP server started on port 80");
    } else {
        Serial.println("❌ HTTP server start failed");
    }
}

// ═══════════════════════════════════════════════════════════
// 🔌 WiFi Reconnect
// ═══════════════════════════════════════════════════════════

void reconnectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;

    Serial.println("🔄 Reconnecting WiFi...");
    WiFi.disconnect();
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);  // ⚡ CRITICAL: No WiFi sleep
    WiFi.begin(ssid, password);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        digitalWrite(LED_GPIO_NUM, !digitalRead(LED_GPIO_NUM));
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n✅ WiFi reconnected!");
        digitalWrite(LED_GPIO_NUM, LOW);
    } else {
        Serial.println("\n❌ WiFi reconnect failed — restarting");
        delay(3000);
        ESP.restart();
    }
}

// ═══════════════════════════════════════════════════════════
// 🚀 SETUP
// ═══════════════════════════════════════════════════════════

void setup() {
    Serial.begin(115200);
    delay(500);

    // ═══════════════════════════════════════════════
    // BOOT BANNER
    // ═══════════════════════════════════════════════
    Serial.println("\n╔══════════════════════════════════╗");
    Serial.println("║  SMART PLAYGROUND MONITOR v2.0   ║");
    Serial.println("║  ESP32-CAM — AI Optimized        ║");
    Serial.println("╚══════════════════════════════════╝\n");

    // LED setup
    pinMode(LED_GPIO_NUM, OUTPUT);
    digitalWrite(LED_GPIO_NUM, LOW);

    // ═══════════════════════════════════════════════
    // CAMERA INIT
    // ═══════════════════════════════════════════════
    if (!initCamera()) {
        Serial.println("❌ Camera init failed — restarting");
        for (int i = 0; i < 10; i++) {
            digitalWrite(LED_GPIO_NUM, HIGH); delay(100);
            digitalWrite(LED_GPIO_NUM, LOW);  delay(100);
        }
        ESP.restart();
        return;
    }

    // ═══════════════════════════════════════════════
    // WiFi CONNECT — NO SLEEP
    // ═══════════════════════════════════════════════
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);  // ⚡ CRITICAL: Disable WiFi sleep
    WiFi.begin(ssid, password);

    Serial.printf("📡 Connecting to WiFi: %s\n", ssid);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
        delay(500);
        Serial.print(".");
        digitalWrite(LED_GPIO_NUM, !digitalRead(LED_GPIO_NUM));
        attempts++;
    }

    digitalWrite(LED_GPIO_NUM, LOW);

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\n❌ WiFi failed — restarting");
        delay(3000);
        ESP.restart();
        return;
    }

    // ═══════════════════════════════════════════════
    // CONNECTION SUCCESS
    // ═══════════════════════════════════════════════
    String ip = WiFi.localIP().toString();

    Serial.println("\n═══════════════════════════════════════");
    Serial.print("✅ IP Address : "); Serial.println(ip);
    Serial.print("📶 Signal   : "); Serial.print(WiFi.RSSI()); Serial.println(" dBm");
    Serial.print("🧠 PSRAM    : "); Serial.println(psramFound() ? "YES" : "NO");
    Serial.print("📦 Frame    : "); Serial.println(psramFound() ? "VGA 640x480" : "CIF 400x296");
    Serial.print("🎨 Quality  : "); Serial.println(jpeg_quality);
    Serial.println("═══════════════════════════════════════");
    Serial.println("\n📡 Endpoints:");
    Serial.print("  Stream : http://"); Serial.print(ip); Serial.println("/stream");
    Serial.print("  Capture: http://"); Serial.print(ip); Serial.println("/capture");
    Serial.print("  Status : http://"); Serial.print(ip); Serial.println("/status");
    Serial.print("  Quality: http://"); Serial.print(ip); Serial.println("/quality?val=10");
    Serial.println("\n🎛️ Quality Control:");
    Serial.println("  ?val=10    (JPEG quality 2-63, 10=recommended)");
    Serial.println("  ?bright=0  (brightness -2 to 2)");
    Serial.println("  ?contrast=0 (contrast -2 to 2)");
    Serial.println("  ?sharp=1   (sharpness 0-2)");
    Serial.println("  ?flip=1    (vertical flip)");
    Serial.println("  ?mirror=1  (mirror)");
    Serial.println("═══════════════════════════════════════\n");

    // ═══════════════════════════════════════════════
    // START SERVER
    // ═══════════════════════════════════════════════
    startServer();

    // ✅ Ready — 3 slow blinks
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_GPIO_NUM, HIGH); delay(400);
        digitalWrite(LED_GPIO_NUM, LOW);  delay(400);
    }

    Serial.println("✅ ESP32-CAM READY — Streaming AI-Optimized Video\n");
}

// ═══════════════════════════════════════════════════════════
// 🔄 LOOP
// ═══════════════════════════════════════════════════════════

void loop() {
    // WiFi monitoring — auto reconnect
    if (WiFi.status() != WL_CONNECTED) {
        reconnectWiFi();
    }

    // Status print every 30 seconds
    static unsigned long lastPrint = 0;
    if (millis() - lastPrint > 30000) {
        Serial.printf(
            "[%lus] RSSI:%ddBm Heap:%d Frames:%lu\n",
            millis() / 1000,
            WiFi.RSSI(),
            ESP.getFreeHeap(),
            frameCount
        );
        lastPrint = millis();
    }

    delay(100);
}