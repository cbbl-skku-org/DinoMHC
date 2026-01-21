import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import math
import os

def calculate_metrics(df, score_col='score_avg', threshold=0.5):
    # Calculate Sensitivity, Specificity, Precision, F1 Score, MCC, AUROC, AUPRC
    y_true = df['label']
    y_scores = df[score_col]
    y_pred = (y_scores >= threshold).astype(int)
    
    tn, fp, fn, tp = [x.item() for x in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()] # Convert numpy types to native Python types
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('inf')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('inf')
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('inf')
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else float('inf')
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else float('inf')
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    
    # Calculate PPV@(number of positives)
    num_positives = sum(y_true)
    if num_positives > 0:
        top_indices = y_scores.nlargest(num_positives).index
        ppv_at_positives = sum(y_true.loc[top_indices]) / num_positives
    else:
        ppv_at_positives = float('inf')
    
    return {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1': f1,
        'MCC': mcc,
        'AUROC': auroc,
        'AUPRC': auprc,
        'PPV@(num_pos)': ppv_at_positives,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }

def main(args):
    paths = args.paths
    dfs = [pd.read_csv(path, sep='\t') for path in paths]

    # Get common columns (all columns except 'score')
    common_cols = [col for col in dfs[0].columns if col != 'score']

    # Start with the first dataframe and rename its score column
    combined = dfs[0].copy()
    combined = combined.rename(columns={'score': 'score_0'})

    # Merge subsequent dataframes
    for i, df in enumerate(dfs[1:], start=1):
        df_renamed = df.rename(columns={'score': f'score_{i}'})
        combined = combined.merge(df_renamed, on=common_cols, how='outer')

    combined['score_avg'] = combined[[f'score_{i}' for i in range(len(dfs))]].mean(axis=1)
    
    # Check if output folder path exists, if not create it
    os.makedirs(args.output_folder_path, exist_ok=True)
    
    # Special case, if users want to group by peptide_len
    if args.groupby_cols and 'peptide_len' in args.groupby_cols:
        combined['peptide_len'] = combined['peptide'].str.len()
        
    # Save combined dataframe
    combined.to_csv(os.path.join(args.output_folder_path, "predictions.tsv"), sep='\t', index=False)
    
    if args.groupby_cols:
        grouped = combined.groupby(by=args.groupby_cols)
        grouped_metrics = {}
        
        for name, group in grouped:
            metrics = calculate_metrics(group, score_col='score_avg', threshold=0.5)
            if len(name) == 1:
                name = name[0]
            else:
                name = '_'.join(map(str, name))
            grouped_metrics[name] = metrics
        
        metrics_df = pd.DataFrame.from_dict(grouped_metrics, orient='index')
        
        # Add average metrics row at the end
        avg_metrics = metrics_df.mean(numeric_only=True)
        avg_metrics.name = 'Average'
        metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_metrics])])
        
        metrics_df.to_csv(os.path.join(args.output_folder_path, "metrics.tsv"), sep='\t')
    else:
        metrics = calculate_metrics(combined, score_col='score_avg', threshold=0.5)
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(args.output_folder_path, "metrics.tsv"), sep='\t', index=False)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument("--paths", nargs='+', required=True, help="Paths to the input TSV files")
    parser.add_argument("--groupby_cols", nargs='+', help="Calculate metrics in each group then averaging", default=None)
    parser.add_argument("--output_folder_path", type=str, help="Path to save the output TSV file", default="./output")
    
    args = parser.parse_args()
    
    main(args)