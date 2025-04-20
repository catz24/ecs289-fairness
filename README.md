# Fair Credit Risk Prediction

This project explores fairness-aware machine learning models for credit risk assessment. Traditional credit scoring algorithms often exhibit bias across demographic groups. In this project, I investigate multiple fairness intervention techniques to mitigate such disparities while maintaining acceptable levels of predictive performance.

## Project Overview

The objective is to evaluate and compare baseline and fairness-aware models across four demographic groups:
- Young Women
- Old Women
- Young Men
- Old Men

Three fairness interventions were implemented and tested:
1. **Demographic Parity**
2. **Equalized Odds**
3. **Mixup-based Fairness Strategy**

## Results Summary

| Group        | Baseline Approval | Demographic Parity | Equalized Odds | Mixup |
|--------------|-------------------|--------------------|----------------|--------|
| Young Women  | 74%               | 81%                | 81%            | 65%    |
| Old Women    | 86%               | 81%                | 81%            | 80%    |
| Young Men    | 80%               | 84%                | 84%            | 73%    |
| Old Men      | 86%               | 82%                | 82%            | 83%    |

- **Demographic Parity**: Achieved the most equal approval rates but at the cost of decreased model discriminative ability (ROC-AUC dropped from ~0.83 to ~0.65).
- **Equalized Odds**: Balanced TPR and FPR across groups, providing a more nuanced fairness-performance tradeoff.
- **Mixup**: Provided stable accuracy (~71%) but failed to improve outcomes for the most disadvantaged group (young women).

## Methodology

- The dataset was preprocessed and split by demographic group.
- Models were trained using `scikit-learn` and evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Fairness constraints were implemented using appropriate libraries and custom logic.
- Results were compared both quantitatively and visually.


## Key Takeaways

- Significant disparities exist in baseline model outcomes for different demographic groups.
- Fairness-aware methods can mitigate bias, but each comes with trade-offs.
- Model selection should be context-dependent—prioritizing fairness, accuracy, or balance depending on application needs.

## Technologies Used

- Python
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn


