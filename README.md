#all test
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import anderson
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def perform_normality_tests(df, columns):
    test_results = {}

    for col in columns:
        column_data = df[col].dropna()  # Remove NaN values before testing

        test_results[col] = {}

        # Shapiro-Wilk Test for Normality
        shapiro_stat, shapiro_p = stats.shapiro(column_data)
        if shapiro_p < 0.05:
            shapiro_inference = 'Reject null hypothesis: Data is not normally distributed.'
        else:
            shapiro_inference = 'Fail to reject null hypothesis: Data is normally distributed.'
        test_results[col]['Shapiro-Wilk Test'] = {'statistic': shapiro_stat, 'p-value': shapiro_p, 'inference': shapiro_inference}

        # Anderson-Darling Test for Normality
        ad_result = anderson(column_data, dist='norm')
        ad_inference = []
        for i, crit_value in enumerate(ad_result.critical_values):
            if ad_result.statistic > crit_value:
                ad_inference.append(f'Reject null hypothesis at {ad_result.significance_level[i]}% significance level.')
            else:
                ad_inference.append(f'Fail to reject null hypothesis at {ad_result.significance_level[i]}% significance level.')
        test_results[col]['Anderson-Darling Test'] = {
            'statistic': ad_result.statistic,
            'critical_values': ad_result.critical_values,
            'inference': ad_inference
        }

        # Kolmogorov-Smirnov Test for Normality
        ks_stat, ks_p = stats.kstest(column_data, 'norm', args=(np.mean(column_data), np.std(column_data)))
        if ks_p < 0.05:
            ks_inference = 'Reject null hypothesis: Data is not normally distributed.'
        else:
            ks_inference = 'Fail to reject null hypothesis: Data is normally distributed.'
        test_results[col]['Kolmogorov-Smirnov Test'] = {'statistic': ks_stat, 'p-value': ks_p, 'inference': ks_inference}

    return pd.DataFrame(test_results)

# Example usage:
# df is your DataFrame, columns is the list of columns you want to test
normality_test_results = perform_normality_tests(df, ['column1', 'column2'])
print(normality_test_results.T)  # Transpose to view results column-wise

#kruskal walis test
import pandas as pd
from scipy import stats

def perform_kruskal_wallis_test(df, group_col, value_col):
    # Group the data by the specified group column
    groups = df.groupby(group_col)[value_col].apply(list).values

    # Check if there are at least two groups with more than one value
    if len(groups) > 1:
        kw_stat, kw_p = stats.kruskal(*groups)
        if kw_p < 0.05:
            kw_inference = 'Reject null hypothesis: Groups are significantly different.'
        else:
            kw_inference = 'Fail to reject null hypothesis: Groups are not significantly different.'
        result = {'Kruskal-Wallis H Test': {'statistic': kw_stat, 'p-value': kw_p, 'inference': kw_inference}}
    else:
        result = {'Kruskal-Wallis H Test': 'Not enough groups for Kruskal-Wallis test.'}

    return result

# Example usage:
# df is your DataFrame, 'group_col' is the column representing group categories, and 'value_col' is the numerical data
kruskal_wallis_result = perform_kruskal_wallis_test(df, 'group_col', 'value_col')
print(kruskal_wallis_result)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer

# Function to apply appropriate transformation based on skewness
def treat_non_normal_distribution(df, columns):
    transformed_columns = {}
    
    for col in columns:
        skewness = df[col].skew()
        plt.figure(figsize=(14, 6))

        # Plot the original distribution
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Original Distribution: {col} (Skewness: {skewness:.2f})')

        # Apply appropriate transformation based on skewness
        if skewness > 0.5:  # Positively skewed
            if all(df[col] > 0):  # Ensure positive values for Box-Cox and Log transformation
                # Try Box-Cox transformation
                transformed_col, fitted_lambda = stats.boxcox(df[col])
                transformation = 'Box-Cox'
            else:
                # Apply Yeo-Johnson transformation for negative and positive data
                pt = PowerTransformer(method='yeo-johnson')
                transformed_col = pt.fit_transform(df[[col]]).flatten()
                transformation = 'Yeo-Johnson'
        elif skewness < -0.5:  # Negatively skewed
            # Apply Exponential transformation to adjust left-skewed data
            transformed_col = np.expm1(df[col].max() - df[col] + 1)
            transformation = 'Exponential'
        else:
            transformed_col = df[col]  # No transformation needed
            transformation = 'No Transformation'

        # Plot the transformed distribution
        plt.subplot(1, 2, 2)
        sns.histplot(transformed_col, kde=True, bins=30)
        plt.title(f'Transformed Distribution ({transformation}): {col}')
        
        # Save the transformed column
        transformed_columns[col] = transformed_col
        
        plt.show()

    return pd.DataFrame(transformed_columns)

# Example usage
# df is your DataFrame and columns contains the columns you want to check
transformed_df = treat_non_normal_distribution(df, ['column1', 'column2', 'column3'])
