# GTA SA Ped Customizer (Streamlit)

One-shot Streamlit app to merge:
- `head.dff`
- `head.txd`
- `body.package` (Sims 4 DBPF)

into a downloadable `final_ped.zip` containing a fixed GTA SA ped DFF/TXD pair.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What this app enforces

- RW version set to `3.6.0.3 (0x1803FFFF)`.
- GTA SA ped hierarchy rewritten with root `BoneID: 0`, `HAnimID: 0x11e`.
- Vertex weight safety: vertices without weights are auto-bound to root with weight `1.0`.
- Body normalization to head bounding box.
- Neck boundary snapping with tolerance `0.0001`.
