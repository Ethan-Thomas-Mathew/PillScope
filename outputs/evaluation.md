# Evaluation

Database: **251 pills**.

### A) Synthetic damage on 251 database photos

| Test | Top-1 | Top-5 | Seconds / photo |
|---|---|---|---|
| unchanged | 100.0% | 100.0% | 0.08 |
| rotated 30 | 99.6% | 99.6% | 0.08 |
| rotated 45 | 99.2% | 99.6% | 0.08 |
| rotated 180 | 99.6% | 99.6% | 0.09 |
| shrunk to 70% | 99.6% | 99.6% | 0.08 |
| gaussian noise | 99.6% | 100.0% | 0.09 |
| darker | 100.0% | 100.0% | 0.07 |

### B) Real 30-degree rotated photos from the dataset

227 `(N)r30.jpg` photos are an original rotated by 30 degrees (checked pixel by pixel). Only 75 of them carry the same number as their original, so the right answer is taken from the pixel check, not the filename.

| Top-1 | Top-5 | Seconds / photo |
|---|---|---|
| 99.1% | 100.0% | 0.09 |
