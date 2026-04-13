import numpy as np
import pandas as pd
import joblib

from src.config import *



class CostPredictor:
    '''production ready cost prediction service'''

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.feature_names = joblib.load(MODEL_PATH.parent / 'feature_names.pkl')


    def get_cost_tier(self, cost):
        '''determine cost tier'''
        if cost < 5000:
            return 'Low', 'Routine care, minimal intervention needed'
        elif cost < 15000:
            return 'Medium', 'Monitor regularly, preventive care recommended'
        elif cost < 30000:
            return 'High', 'Active case managment recommended'
        else:
            return 'Very High', 'Immediate intervention required'


    def get_recommendations(self, cost_tier, smoker, age, bmi):
        '''generate actionable recommendations'''
        recommendations = []
        if cost_tier in ['High', 'Very High']:
            recommendations.append('Schedule care management sconsulation')

        if smoker == 'yes':
            recommendations.append('Enroll in smoking cessation program')

        if age > 50 and bmi > 30:
            recommendations.append('Comprehensive health assessment needed')
        elif bmi > 30:
            recommendations.append('Weight management program recommended')

        if not recommendations:
            recommendations.append('Continue routine preventive care')


        return recommendations


    def predict(self, age, sex, bmi, children, smoker, region):
        '''
        predict healthcare consts for a patient
        parameters:
        - age: int(18-100)
        - sex: str ('male' or 'female')
        - bmi: float(15-50)
        - childeren: int(0-10)
        - smoker: str ('yes' or 'no')
        - region: str ('norheast', 'nortwest', 'southeast', 'southwest')

        returns:
        - dict with prediciton, confidences, recommentations
        '''

        # create feature vector
        # calculate interaction features
        bmi_age_interaction = bmi * age
        smoker_age = 1 if smoker == 'yes' else 0 * age
        children_smoker = (1 if smoker == 'yes' else 0) * children

        'encode categorical'
        sex_male = 1 if sex == 'male' else 0
        smoker_yes = 1 if smoker == 'yes' else 0

        # region one-hot (assuming notheast is hbaseline)
        region_northwest = 1 if region == 'northwest' else 0
        region_southeast = 1 if region == 'southeast' else 0
        region_southwest = 1 if region == 'southwest' else 0

        features = pd.DataFrame([[
            age, bmi, children,
            bmi_age_interaction, smoker_age, children_smoker,
            sex_male, smoker_yes,
            region_northwest, region_southeast, region_southwest
        ]], columns=self.feature_names)


        # predict
        predicted_cost = self.model.predict(features)[0]

        # get tier
        cost_tier, clinical_note = self.get_cost_tier(predicted_cost)

        # get_recommendations
        recommendation = self.get_recommendations(cost_tier, smoker, age, bmi)

        # calculate confidence interval (simplified - based on tpical model error)
        confidence_interval = (predicted_cost * 0.85, predicted_cost * 1.15)

        return {
            'predicted_cost' : round(predicted_cost, 2),
            'cost_tier' : cost_tier,
            'clinical_note': clinical_note,
            'confidence_interval': [round(confidence_interval[0], 2), round(confidence_interval[1], 2)],
            'recommendations': recommendation,
            'risk_factors': {
                'smoker': smoker == 'yes',
                'high_bmi': bmi > 30,
                'senior': age > 60,
            }
        }


# singleton instance
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = CostPredictor()
    return _predictor


if __name__ == '__main__':
    # test the predictor
    predictor = get_predictor()

    # test case 1: young non-smoker
    result = predictor.predict(
        age=25, sex='female', bmi=22.5, children=0,
        smoker='no', region='northeast'
    )

    print('\ntest case 1: Young non-smoker')
    print(result)

    # test case 2: older smoker
    result = predictor.predict(
        age=55, sex='male', bmi=32.0, children=2,
        smoker='yes', region='southeast'
    )

    print('\ntest case 2: Older smoker with high BMI')
    print(result)
