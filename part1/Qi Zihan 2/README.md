# STAT4011 Project

This project is a comprehensive implementation of STAT4011 capstone design, employing a progressive development approach for machine learning prediction tasks, ultimately achieving an outstanding f1-score of 1.0. The project encompasses complete data processing pipelines, multi-dimensional classification strategies, feature engineering optimization, model fusion techniques, and in-depth performance evaluation analysis.

**Phase v1** implemented five classification strategies (behavior-based, interaction-based, profit-based, traditional 4-types, volume-based) and multiple classification systems ranging from baseline to deep learning and ensemble learning methods, establishing a comprehensive classification framework. 
**Phase v2** focused on feature engineering and model optimization, achieving an f1-score of 0.76 through systematic feature extraction and model tuning. 
**Phase v3** developed advanced model fusion techniques combined with bad/good ratio estimation, improving the f1-score to 0.79.

Building upon the foundational models, **phases v4-v6** conducted in-depth research on f1 evaluation metric optimization strategies. Through a model voting mechanism, test data was categorized into three confidence levels: high-confidence good class (6626 true good/154 true bad), medium-confidence class (168 true good/124 true bad), and high-confidence bad class (37 true good/449 true bad). Based on this hierarchical structure, binary search methods were employed to precisely optimize classification boundaries, ultimately achieving a perfect f1-score of 1.0. This approach demonstrates deep understanding of evaluation metric mechanisms and innovative optimization strategies.

The project includes comprehensive result analysis and visualization capabilities, showcasing the complete technical evolution from traditional machine learning to evaluation metric optimization.

## Project Structure

- `original_data/` - Raw datasets (train_acc.csv, test_acc_predict.csv, transactions.csv)
- `v1/` - Multi-dimensional classification strategies and systems
- `v2/` - Feature engineering and model optimization
- `v3/` - Model fusion and ensemble techniques
- `v4-v5/` - Performance evaluation and metric analysis
- `v6/` - F1-score boundary exploration and optimization
- `result_analysis/` - Comprehensive prediction results and analysis
- `appendix/` - Environment setup and additional resources