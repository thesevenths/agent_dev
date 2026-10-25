import json, glob, statistics as st
tk=[]; dur=[]
for f in glob.glob(r"experiments/calibration/20261023T113948/*/*/result.json"):
    d=json.load(open(f,encoding="utf-8"))
    if d.get("ok"):
        tk.append(d.get("tokens") or 0)
for f in glob.glob(r"experiments/calibration/20261023T113948/*/*/outcome.json"):
    d=json.load(open(f,encoding="utf-8"))
    if d.get("duration_s"): dur.append(d["duration_s"])
print("ok runs:", len(tk), "tokens avg:", int(st.mean(tk)) if tk else None, "min/max:", (min(tk),max(tk)) if tk else None)
print("durations n=", len(dur), "avg:", round(st.mean(dur),1) if dur else None, "max:", max(dur) if dur else None)
