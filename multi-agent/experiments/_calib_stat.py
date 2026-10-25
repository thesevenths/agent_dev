import json, glob, statistics as st
rows=[]
for f in glob.glob(r"experiments/calibration/20261023T113948/*/*/result.json"):
    d=json.load(open(f,encoding="utf-8"))
    rows.append((f.split("\\")[-3], d.get("ok"), d.get("Q"), d.get("tokens")))
print("runs:",len(rows))
bad=[r for r in rows if not r[1]]
print("not_ok:",len(bad), bad[:5])
q=[r[2] for r in rows if r[1]]
print("Q min/median/max:", min(q), st.median(q), max(q) if q else None)
tok=[r[3] for r in rows]
print("tokens total:", sum(tok), "avg:", int(st.mean(tok)))
