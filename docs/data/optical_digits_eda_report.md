# Optical Digits EDA Report

## Overview
- Rows: 5620
- Features: 64
- Target: class
- Task type: 10-class handwritten digit classification
- Image structure: 8x8 grid flattened into 64 integer features
- Pixel value range: 0 to 16 on-pixel counts per 4x4 block
- Source: UCI Optical Recognition of Handwritten Digits dataset

## How To Read This Dataset
- Each row is one handwritten digit sample represented as 64 block-intensity features.
- The 64 columns can be reshaped into an 8x8 image for CNN input or visual inspection.
- All features are already numeric, bounded, and free of missing values.
- The classes are close to balanced, so only light class-weight correction is typically needed.
- Feature scaling is straightforward: divide pixel counts by 16 to normalize inputs to [0, 1].

## Target Distribution
```text
       count  percentage
class                   
0        554        9.86
1        571       10.16
2        557        9.91
3        572       10.18
4        568       10.11
5        558        9.93
6        558        9.93
7        566       10.07
8        554        9.86
9        562       10.00
```

## Data Quality
- Missing feature values: 0
- Missing target values: 0
- Duplicate feature rows: 0
- Duplicate full rows: 0
- Constant pixels: 2 (Attribute1, Attribute40)

```text
             min  max  mean  std
Attribute1     0    0   0.0  0.0
Attribute40    0    0   0.0  0.0
```

## Pixel Summary Highlights
```text
             min  max   mean   std
Attribute60    0   16  11.99  4.35
Attribute4     0   16  11.82  4.26
Attribute12    0   16  11.80  4.00
Attribute5     0   16  11.58  4.46
Attribute61    0   16  11.57  4.98
Attribute11    0   16  10.51  5.43
Attribute13    0   16  10.51  4.79
Attribute37    0   16  10.33  5.92
Attribute52    0   16   9.77  5.17
Attribute29    0   16   9.75  6.24
```

- Higher-mean pixels trace the central stroke regions where handwritten digits overlap most often.
- Border pixels are sparser, which is expected after the original 32x32 images were block-compressed to 8x8.
- A couple of positions are constant zeroes across the dataset, so they can be dropped without losing signal.

## Average Image Across All Digits
```text
       col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
row_0   0.00   0.30   5.39  11.82  11.58   5.59   1.38   0.14
row_1   0.00   1.97  10.51  11.80  10.51   8.26   2.09   0.14
row_2   0.00   2.60   9.68   6.82   7.16   7.97   1.96   0.05
row_3   0.00   2.38   9.19   9.03   9.75   7.77   2.33   0.00
row_4   0.00   2.14   7.66   9.18  10.33   9.05   2.91   0.00
row_5   0.02   1.46   6.59   7.20   7.84   8.53   3.49   0.02
row_6   0.01   0.78   7.75   9.77   9.65   9.12   3.74   0.17
row_7   0.00   0.28   5.76  11.99  11.57   6.72   2.09   0.25
```

## Digit-Wise Intensity Summary
```text
       count  avg_total_ink  std_total_ink
class                                     
0        554         319.80          35.54
1        571         322.80          45.80
2        557         309.29          31.22
3        572         307.29          31.35
4        568         309.96          29.34
5        558         304.60          31.58
6        558         309.56          30.94
7        566         306.41          28.73
8        554         336.46          38.09
9        562         317.36          35.96
```

## Average Pixel Grid By Digit
### Digit 0
```text
       col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
row_0    0.0   0.03   4.44  13.15  11.06   2.73   0.05    0.0
row_1    0.0   1.08  12.85  13.19  12.03  10.93   0.81    0.0
row_2    0.0   3.83  14.34   4.88   2.71  12.55   3.39    0.0
row_3    0.0   5.25  12.74   1.77   0.27   9.17   6.57    0.0
row_4    0.0   5.57  11.80   0.93   0.14   8.70   7.30    0.0
row_5    0.0   3.37  13.45   1.75   1.48  11.36   6.26    0.0
row_6    0.0   0.77  13.10  10.38  10.55  13.41   2.60    0.0
row_7    0.0   0.01   4.44  13.56  13.32   5.52   0.23    0.0
```
### Digit 1
```text
       col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
row_0   0.00   0.01   2.38   9.08  10.86   5.87   0.67   0.00
row_1   0.00   0.18   4.55  12.66  14.55   8.99   1.00   0.00
row_2   0.03   1.32   7.75  14.73  14.68   7.92   0.60   0.00
row_3   0.01   2.35   9.78  14.55  14.42   7.10   0.43   0.00
row_4   0.00   1.48   7.26  12.01  14.24   6.72   0.39   0.00
row_5   0.00   0.40   4.86  10.46  13.96   7.14   0.49   0.00
row_6   0.00   0.13   4.56  10.94  14.18   8.95   1.77   0.33
row_7   0.00   0.03   2.30   8.80  12.71   8.86   2.56   0.82
```
### Digit 2
```text
       col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
row_0   0.00   0.94  10.13  14.03   8.25   1.70   0.10   0.00
row_1   0.00   5.06  14.36  12.46  12.27   4.66   0.32   0.00
row_2   0.00   4.46   8.02   4.35  11.28   5.61   0.38   0.00
row_3   0.00   1.04   2.15   4.65  11.63   4.44   0.17   0.00
row_4   0.00   0.07   1.22   8.34  10.29   2.26   0.06   0.00
row_5   0.00   0.36   4.87  11.56   7.16   1.62   0.53   0.01
row_6   0.01   1.50  11.65  14.52  12.04  10.65   6.94   0.47
row_7   0.00   0.96  10.41  14.02  13.18  12.06   8.62   1.44
```
### Digit 3
```text
       col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
row_0    0.0   0.76   8.65  14.01  13.50   6.22   0.72   0.03
row_1    0.0   4.24  12.90   9.21  11.72  11.17   1.81   0.03
row_2    0.0   1.88   3.45   3.23  11.76   9.13   0.77   0.00
row_3    0.0   0.19   1.94  10.03  14.24   5.90   0.18   0.00
row_4    0.0   0.06   1.47   7.02  12.32  11.73   2.43   0.00
row_5    0.0   0.17   0.78   0.76   4.03  12.60   6.58   0.01
row_6    0.0   0.89   7.26   6.39   8.30  13.53   5.88   0.04
row_7    0.0   0.64   9.58  14.56  13.34   7.96   1.29   0.02
```
### Digit 4
```text
       col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
row_0   0.00   0.01   0.78   6.52  11.20   2.47   0.51   0.19
row_1   0.00   0.13   3.26  12.27   8.78   2.46   2.13   0.40
row_2   0.00   0.80   9.20  11.66   4.33   6.04   4.44   0.30
row_3   0.00   3.99  13.77   8.11   6.16  11.48   6.21   0.02
row_4   0.01   7.19  14.38  10.14  12.23  14.21   4.98   0.00
row_5   0.21   6.02  10.33  11.51  14.58  10.90   1.67   0.00
row_6   0.13   1.65   3.35   7.71  13.55   5.14   0.27   0.01
row_7   0.00   0.05   0.89   7.26  11.46   2.39   0.07   0.01
```
### Digit 5
```text
       col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
row_0   0.00   0.65   8.64  13.01  13.98  11.57   3.59   0.09
row_1   0.01   3.30  14.35  12.56   9.07   7.29   2.18   0.03
row_2   0.01   4.81  14.11   6.37   2.39   0.72   0.07   0.00
row_3   0.00   4.75  13.84  12.13   8.97   4.36   0.40   0.00
row_4   0.00   1.51   6.75   7.85   8.78   7.93   1.88   0.00
row_5   0.00   0.15   0.96   2.86   6.84   9.08   2.54   0.00
row_6   0.00   0.72   6.11   8.08  10.98   8.55   1.60   0.01
row_7   0.00   0.57   9.41  14.47  10.17   3.21   0.36   0.00
```
### Digit 6
```text
       col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
row_0    0.0   0.01   1.79  10.92   8.51   1.23   0.00   0.00
row_1    0.0   0.13   7.76  13.90   5.81   0.85   0.02   0.00
row_2    0.0   0.91  12.58   9.00   0.73   0.05   0.00   0.00
row_3    0.0   2.25  13.78   7.29   3.09   1.37   0.08   0.00
row_4    0.0   3.39  14.59  12.24  11.70   9.66   2.30   0.00
row_5    0.0   2.05  14.77  11.39   6.94  10.77   9.15   0.20
row_6    0.0   0.36  10.58  12.80   6.06  11.14  11.16   0.72
row_7    0.0   0.01   1.90  10.60  14.79  12.90   5.15   0.20
```
### Digit 7
```text
       col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
row_0    0.0   0.32   6.39  13.26  14.36  11.92   5.91   0.99
row_1    0.0   0.97   9.39  10.84  11.06  13.06   6.77   0.73
row_2    0.0   0.53   2.55   1.32   6.74  11.99   3.94   0.12
row_3    0.0   0.50   3.69   6.69  12.49  12.34   4.44   0.00
row_4    0.0   1.51   8.77  13.80  15.13  12.20   4.71   0.00
row_5    0.0   0.91   4.51  12.09  10.72   3.83   0.69   0.00
row_6    0.0   0.13   4.23  13.11   5.30   0.23   0.00   0.00
row_7    0.0   0.31   7.67  11.35   1.79   0.08   0.00   0.00
```
### Digit 8
```text
       col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
row_0   0.00   0.14   5.39  12.33  12.36   6.05   0.49   0.00
row_1   0.02   2.31  13.10  10.59   9.29  11.83   2.43   0.00
row_2   0.00   3.51  12.17   6.60   8.07  11.55   2.18   0.00
row_3   0.00   1.34   9.61  13.50  13.44   6.90   0.60   0.00
row_4   0.00   0.40   7.61  14.33  12.25   4.42   0.36   0.00
row_5   0.00   1.11  11.27   8.85   8.87   7.93   1.66   0.00
row_6   0.00   1.17  12.01   8.30   8.20   9.85   2.58   0.00
row_7   0.00   0.13   5.58  13.18  12.79   6.78   1.02   0.02
```
### Digit 9
```text
       col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
row_0    0.0   0.15   5.41  11.95  11.63   6.10   1.73   0.08
row_1    0.0   2.35  12.87  10.38  10.49  11.30   3.34   0.16
row_2    0.0   4.05  13.00   5.92   8.64  14.10   3.86   0.07
row_3    0.0   2.19  10.77  11.47  12.57  14.49   4.24   0.00
row_4    0.0   0.21   2.85   5.08   5.99  12.49   4.71   0.00
row_5    0.0   0.09   0.39   0.75   3.62  10.07   5.41   0.01
row_6    0.0   0.52   5.00   5.60   7.31   9.84   4.68   0.08
row_7    0.0   0.11   5.47  12.25  12.26   7.53   1.71   0.03
```

## Recommended Report Extensions
- Add rendered heatmaps for the average 8x8 image overall and per digit class.
- Compare confusion-prone pairs like 3 vs 5 or 8 vs 9 after a first baseline model run.
- Track validation performance on the predefined digit classes to confirm stratified splits stay balanced.
- Show example reconstructions from low-ink and high-ink samples to catch preprocessing mistakes early.
- Document the final tensor layout clearly, especially whether the CNN expects channel-last or channel-first input.
