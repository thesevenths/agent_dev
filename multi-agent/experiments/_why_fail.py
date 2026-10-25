import json, glob
for f in glob.glob(r"experiments/calibration/20261023T113948/*/*/result.json"):
    d=json.load(open(f,encoding="utf-8"))
    if not d.get("ok"):
        print("==", f.split("113948")[1])
        print("   keys:", sorted(d))
        print("   error:", str(d.get("error"))[:300])
        print("   infra:", d.get("infra_error"), "calls:", d.get("calls"), "errs:", d.get("call_errors"))
