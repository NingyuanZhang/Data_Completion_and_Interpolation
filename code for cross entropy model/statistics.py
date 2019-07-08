import pandas as pd
from dataloader import DataLoader
import os

def main():
    source_data_path = '../data/data_all.csv'
    predicted_data_path = '../output/predicted_data_all.csv'
    header_file_path = '../data/new_features.csv'
    output_path = '../output/statistic_data_all.csv'
    # source_data_path = 'data/data_all.csv'
    # total_header_file_path = 'data/data_all.csv'
    # predicted_data_path = 'output/predicted_simulate_data.csv'
    # header_file_path = 'data/new_features.csv'
    # output_path = 'output/statistic_simulate_data.csv'

    if os.path.exists(source_data_path) is False:
        raise Exception('Source data file not found')
    if os.path.exists(predicted_data_path) is False:
        raise Exception('Prediction data file not found')

    # Load data
    source_df = pd.read_csv(source_data_path, sep=',')
    # make sure to skip the second row of predicted data file which contains the header
    predicted_df= pd.read_csv(predicted_data_path, sep=',', skiprows=[1])

    dl_for_header = DataLoader(source_data_path, header_file_path)
    _, y = dl_for_header.get_xy_headers()

    output_df = None
    for col_name in y:
        series_df1 = source_df[col_name].value_counts(normalize=True) * 100
        series_df1.index = series_df1.index.astype(int)
        series_df1 = series_df1.sort_index()
        tmp_df1 = pd.DataFrame(series_df1)

        series_df2 = predicted_df[col_name].value_counts(normalize=True) * 100
        series_df2.index = series_df2.index.astype(int)
        series_df2 = series_df2.sort_index()
        tmp_df2 = pd.DataFrame(series_df2)

        tmp_df1 = pd.concat([tmp_df1, tmp_df2], axis=1)
        tmp_df1.columns = [[col_name, col_name], ['source', 'predicted']]

        if output_df is None:
            output_df = tmp_df1
        else:
            output_df = output_df.join(tmp_df1, how='outer')
    # print(output_df)
    output_df.to_csv(output_path, sep=',', encoding='utf-8')

if __name__ == '__main__':
    main()
