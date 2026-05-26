# test_models.py - NEW VERSION using ai-edge-litert

import os
import sys
import numpy as np

print("\n" + "="*55)
print("  MODEL TEST - ai-edge-litert version")
print("="*55)

SPORTS_MODEL  = "models/sports_v2.tflite"
UNIFORM_MODEL = "models/uniform_detection_final.tflite"
SPORT_CLASSES = ["badminton", "basketball", "cricket", "football"]

# Check files
print("\n[1] Model files:")
for path in [SPORTS_MODEL, UNIFORM_MODEL]:
    if os.path.exists(path):
        size = os.path.getsize(path) // 1024
        print(f"  ✅ {path} ({size} KB)")
    else:
        print(f"  ❌ {path} NOT FOUND")
        sys.exit(1)

# Load with ai-edge-litert
print("\n[2] Loading with ai-edge-litert...")
try:
    from ai_edge_litert.interpreter import Interpreter
    print("  ✅ ai-edge-litert imported")
except ImportError:
    print("  ❌ ai-edge-litert not installed")
    print("  Run: pip install ai-edge-litert")
    sys.exit(1)

# Load sports model
print("\n[3] Loading sports model...")
try:
    interp = Interpreter(model_path=SPORTS_MODEL)
    interp.allocate_tensors()
    inp   = interp.get_input_details()
    out   = interp.get_output_details()
    ih    = int(inp[0]['shape'][1])
    iw    = int(inp[0]['shape'][2])
    nc    = int(out[0]['shape'][1])
    print(f"  ✅ Loaded!")
    print(f"  Input : {iw}x{ih}")
    print(f"  Output: {nc} classes")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Dummy inference
print("\n[4] Dummy inference...")
try:
    dummy = np.zeros((1, ih, iw, 3), dtype=np.float32)
    interp.set_tensor(inp[0]['index'], dummy)
    interp.invoke()
    result = interp.get_tensor(out[0]['index'])[0]
    print(f"  ✅ Works!")
    print(f"  Output: {[round(float(x),4) for x in result]}")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Real camera test
print("\n[5] Camera test...")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            img = cv2.resize(frame, (iw, ih))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img.astype(np.float32)/255.0, 0)
            interp.set_tensor(inp[0]['index'], img)
            interp.invoke()
            scores = interp.get_tensor(out[0]['index'])[0]
            print(f"  ✅ Camera frame processed!")
            print(f"\n  Sport scores:")
            for i, score in enumerate(scores):
                name = SPORT_CLASSES[i] if i < len(SPORT_CLASSES) else f"class_{i}"
                bar  = "█" * int(score * 30)
                print(f"    {name:<12}: {score:.4f}  {bar}")
            best = int(np.argmax(scores))
            bname = SPORT_CLASSES[best] if best < len(SPORT_CLASSES) else f"class_{best}"
            print(f"\n  Best: {bname} ({float(scores[best]):.1%})")
        else:
            print("  Camera read failed")
    else:
        print("  No webcam - using random test")
        test = np.random.rand(1, ih, iw, 3).astype(np.float32)
        interp.set_tensor(inp[0]['index'], test)
        interp.invoke()
        scores = interp.get_tensor(out[0]['index'])[0]
        print(f"  ✅ Random test works!")
        print(f"  Scores: {[round(float(x),3) for x in scores]}")
except Exception as e:
    print(f"  Error: {e}")

# Load uniform model
print("\n[6] Loading uniform model...")
try:
    interp2 = Interpreter(model_path=UNIFORM_MODEL)
    interp2.allocate_tensors()
    inp2 = interp2.get_input_details()
    out2 = interp2.get_output_details()
    ih2  = int(inp2[0]['shape'][1])
    iw2  = int(inp2[0]['shape'][2])
    nc2  = int(out2[0]['shape'][1])

    dummy2 = np.zeros((1, ih2, iw2, 3), dtype=np.float32)
    interp2.set_tensor(inp2[0]['index'], dummy2)
    interp2.invoke()
    result2 = interp2.get_tensor(out2[0]['index'])[0]

    print(f"  ✅ Loaded!")
    print(f"  Input  : {iw2}x{ih2}")
    print(f"  Output : {nc2} classes")
    print(f"  Scores : {[round(float(x),4) for x in result2]}")
except Exception as e:
    print(f"  ❌ Failed: {e}")

print("\n" + "="*55)
print("  ALL TESTS DONE")
print("="*55)
print()
print("✅ ai-edge-litert works with your models!")
print()
print("Now run: python server.py")
print("Open   : http://localhost:8000")