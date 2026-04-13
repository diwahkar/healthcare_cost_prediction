import pandas as pd


def create_interaction_features(df):
    '''create domain specific interaction features'''
    df_copy = df.copy()

    # bmi*age (older + obese = higher risk)
    df_copy['bmi_age_interaction'] = df_copy['bmi'] * df_copy['age']

    # smoker * age (smoking impact increases with age)
    df_copy['smoker_age'] = (df_copy['smoker'] == 'yes').astype(int) * df_copy['age']

    # children*smoker (family impact of smoking)
    df_copy['children_smoker'] = (df_copy['smoker']=='yes').astype(int) * df_copy['children']


    # bmi category (clinical bins)
    df_copy['bmi_category'] = pd.cut(df_copy['bmi'],
                                bins = [0, 18.5, 25, 30, 100],
                                labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
                              )

    df_copy['age_group'] = pd.cut(df_copy['age'],
                              bins = [0, 30, 40, 50, 60, 100],
                              labels = ['<30', '30-40', '40-50', '50-60', '60+'],
                           )


    return df_copy



def add_cost_tier(df):
    '''add cost tier column for analysis'''
    df_copy = df.copy()


    def get_tier(cost):
        if cost < 5000:
            return 'Low'

        elif cost < 15000:
            return 'Medium'

        elif cost < 30000:
            return 'High'

        else:
            return 'Very High'


    df_copy['cost_tier'] = df_copy['charges'].apply(get_tier)



def get_features_names():
    '''return list of all features after engineering'''
    return [
        'age', 'bmi', 'children',
        'bmin_age_interaction', 'smoker_age', 'children_smoker',
        'sex_male', 'smoker_yes', 'region_nothwest',
        'region_souteast', 'region_soutwest'
    ]


if __name__ == '__main__':
    from src.data_preprocessing import load_data

    df = load_data()
    df = create_interaction_features(df)
    print(f'New features created. Shape: {df.shape}')
    print(f'New columns: {df.columns.tolist()}')
