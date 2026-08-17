# LeRobot Dataset Clone (Educational)

> A clean, minimal, and well-commented re-implementation of the core ideas behind Hugging Face’s [`LeRobotDataset`](https://github.com/huggingface/lerobot).

This project exists to help people **deeply understand** the LeRobot dataset format, by reading and experimenting with a simplified version.

---

### Why this exists

Robot learning datasets are more complex than typical vision or language datasets:

- Multi-modal (images/videos + proprioception + actions)
- Temporal / episodic
- High frequency
- Need to scale to millions of episodes

LeRobot solved this with a clever design:
- Physical storage is **file-size oriented** (chunked Parquet + MP4)
- Logical access is **episode/frame oriented** (reconstructed via rich metadata)

This repository strips away production complexity so you can clearly see how that design works.

---

### What is implemented

- [x] Core dataset class with the same high-level spirit as `LeRobotDataset`
- [x] Loading of `meta/info.json`, episode metadata, and basic tabular data
- [x] Frame indexing (`dataset[idx]`)
- [ ] Support for `delta_timestamps` (temporal context windows)
- [ ] Clear separation between metadata, tabular data, and video frames
- [ ] Extensive comments explaining *why* things are done a certain way

### What is simplified / not implemented

- Dataset writing / recording (`create`, `add_frame`, `save_episode`...)
- Hugging Face Hub download & caching logic
- Advanced features (depth, multi-dataset, streaming, etc.)
- Production-grade performance optimizations

The goal is **clarity**, not completeness.

---

### Quick Start

```bash
git clone https://github.com/laihongphuc/minLerobot.git
cd minLerobot
pip install -e .
