# BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction

This repository contains the official data processing scripts for the paper:

> **BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction**  
> *AAAI Conference on Artificial Intelligence (AAAI-26)*  
> Xu, Z., Cheng, S., Wang, L., He, H., Sun, W., Li, J., & Xu, L. L. (2026)  
> https://doi.org/10.1609/aaai.v40i46.41299  
  [ojs.aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/41299)

---

## 🔥 Overview

Wildfire risk prediction is a complex challenge involving interactions among **fuel**, **meteorology**, **topography**, and **human activity**. Despite increasing interest in data-driven wildfire modeling, **public benchmark datasets** that support:

- long-term temporal modeling  
- large-scale spatial coverage  
- multimodal environmental drivers  

remain extremely limited.

**BCWildfire** addresses this gap by providing:

- **25 years** of wildfire-related data  
- **Daily resolution**  
- **240 million hectares** across British Columbia and surrounding regions  
- **38 multimodal covariates**, including:  
  - active fire detections  
  - weather variables  
  - fuel conditions  
  - terrain features  
  - anthropogenic factors  
  [ojs.aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/41299)

This repository provides the dataset interface, preprocessing tools, and benchmark implementations.

---

## 🧠 Benchmark Models

The paper evaluates a diverse set of time-series forecasting architectures, including:

- **CNN-based models**  
- **Linear models**  
- **Transformer-based models**  
- **Mamba-based architectures**  
  [ojs.aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/41299)

Additionally, the benchmark investigates:

- the effectiveness of **positional embeddings**  
- the relative importance of different **fire-driving factors**

All baseline implementations and training configurations are included in this repo.

---

## 📊 Dataset Description

### Key Properties
| Property | Description |
|---------|-------------|
| Spatial Coverage | 240 million hectares (British Columbia + surrounding regions) |
| Temporal Range | 25 years |
| Temporal Resolution | Daily |
| Covariates | 38 multimodal drivers |
| Data Types | Fire detections, weather, fuel, terrain, human activity |

### Covariate Categories
- **Active fire detections**  
- **Meteorological variables** (temperature, humidity, wind, etc.)  
- **Fuel conditions** (vegetation, drought indices)  
- **Terrain features** (elevation, slope, aspect)  
- **Anthropogenic factors** (roads, population, etc.)

---
## Dataset Downloading
The sampled dataset for model training, validation, and testing can be found under: [huggingface BCWildfire](https://huggingface.co/datasets/RyanSyn/BCWildfire)

---

## Baseline Models
[Github Models](https://github.com/SynUW/hackathon10_WildfireForecasting)

---
## ✍️ Citation

If you use the dataset or code, please cite:

```
@article{xu2026bcwildfire,
  title={BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction},
  author={Xu, Zhengsen and Cheng, Sibo and Wang, Lanying and He, Hongjie and Sun, Wentao and Li, Jonathan and Xu, Lincoln Linlin},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={46},
  pages={39486--39494},
  year={2026},
  doi={10.1609/aaai.v40i46.41299}
}
```

---

## 📬 Contact

For questions or collaborations, please contact the zhengsen.xu@ucalgary.ca or open an issue in this repository.
---
