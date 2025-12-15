# Semantic Enhancement or Regularization?
### Disentangling Performance Gains in Personality-Aware Recommendation Systems

This repository contains the official implementation for the paper:

> **Semantic Enhancement or Regularization? Disentangling Performance Gains in Personality-Aware Recommendation Systems**  
> Abhradeep Datta, Varun Tej Kasula, Ankit Varshney, Ashok Singh Sairam  
> IIT Guwahati

---

## Overview

Personality-aware recommender systems are commonly believed to improve performance by injecting semantic user traits.
This work shows that most observed gains instead arise from **implicit regularization effects**, not semantic information.

We systematically compare:
- Real personality traits
- Shuffled personality traits
- Random uniform noise

across GraphSAGE-based recommendation models using linear fusion, concatenation, and multi-task learning (MTL).

---

## Datasets

- **Personality2018 (MovieLens)**  
  Includes explicit Big Five personality scores.

- **Amazon Music**  
  Personality inferred from review text using SetFit.

