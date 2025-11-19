# A Population-Specific Ensemble Machine Learning Model for Predicting Borderline or Malignancy Risk of Ovarian Masses in Macao: A Multicenter Retrospective Study

Chan-Fong CHIO, et al.

DOI: [10.1177/11795549251388312](https://doi.org/10.1177/11795549251388312)

## 1. Requirements

```textile
python==3.11
scikit_learn==1.3.2
pandas==2.2.2
numpy==1.26.4
shap==0.46.0
joblib==1.4.2
matplotlib==3.7.3
```

## 2. Environment

#### 2.1. Create base environment with python 3.11

```bash
conda create --name sk132 python=3.11
```

#### 2.2. Install packages

```bash
conda activate sk132
pip install -r requirements.txt
```

## 3. Usage

#### 3.1. Input features manaully

```bash
python mo-ovary-predict.py --input="Age:49, Menopause:0, US1:0, US2:1, US3:0, US4:0, US5:0, US6:1, US7:0, CA125:54.8, BUN:3.7, AST:14, ALT:11, WBC:5, Lym:1, Plt:240" --model=model-23-01-sk132.json
```
Output:
```textile
-------------------------------------
Model: mo-ovary-23-01-sk132
-------------------------------------
    Age  Menopause  US1  US2  US5  US6  US7  CA125  BUN   ALT       AAR  WBC  Lym    PLR
0  49.0        0.0  0.0  1.0  0.0  1.0  0.0   54.8  3.7  11.0  1.272727  5.0  1.0  240.0

#0  Borderline/Malignant likelihood = 0.7294 High risk
    RMI-4 = 109.0 Low risk
```

#### 3.2. Input features from a CSV/Excel file

```bash
python mo-ovary-predict.py --input=test.csv --model=model-23-01-sk132.json
```
Output:
```textile
-------------------------------------
Model: mo-ovary-23-01-sk132
-------------------------------------
   Age  Menopause  US1  US2  US5  US6  US7  CA125  BUN  ALT       AAR   WBC  Lym         PLR
0   49          0    0    1    0    1    0   54.8  3.7   11  1.272727   5.0  1.0  240.000000
1   53          1    1    1    0    1    0   93.0  3.0   21  0.857143  12.8  0.8  455.000000
2   35          0    1    1    0    1    0  164.0  2.5   10  1.300000  10.5  3.5  109.142857
3   70          1    0    1    0    1    0   11.4  6.5   24  1.083333   5.4  2.4  102.916667

     ID       VTP  RMI4  VT  R4
0  T001  0.729387   109   1   0
1  T002  0.831473  2976   1   1
2  T003  0.735311  1312   1   1
3  T004  0.621146    91   1   0
```

#### 3.3. Export outputs to a CSV/Excel file

```bash
python mo-ovary-predict.py --input=test.csv --output=result_test.csv --model=model-23-01-sk132.json
```

#### 3.4. Export SHAP forceplot (*only support manual input, i.e. method 3.1*)

```bash
python mo-ovary-predict.py --input="Age:49, Menopause:0, US1:0, US2:1, US3:0, US4:0, US5:0, US6:1, US7:0, CA125:54.8, BUN:3.7, AST:14, ALT:11, WBC:5, Lym:1, Plt:240" --output=shap --model=model-23-01-sk132.json
```
Output:
```textile
-------------------------------------
Model: mo-ovary-23-01-sk132
-------------------------------------
    Age  Menopause  US1  US2  US5  US6  US7  CA125  BUN   ALT       AAR  WBC  Lym    PLR
0  49.0        0.0  0.0  1.0  0.0  1.0  0.0   54.8  3.7  11.0  1.272727  5.0  1.0  240.0

#0  Borderline/Malignant likelihood = 0.7294 High risk
    RMI-4 = 109.0 Low risk

===== [ SHAP Analysis ] =====
100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.80s/it]
==> Output to shap-forceplot.html
```
