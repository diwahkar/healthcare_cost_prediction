import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator

from src.predict import get_predictor



# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# initialize FastAPI
app = FastAPI(
    title='Healthcare Cost Prediction API',
    description='Predict patient healthcare costs to support value-based care decisions',
    versions='1.0.0',
    docs_url='/docs',
    redoc_url='/redocs'
)


# request/response models
class PatientData(BaseModel):
    '''patient health and demographic data'''
    age: int = Field(..., ge=18, le=100, description='age in years (18-100)')
    sex: str = Field(..., desctiption='sex: \'male\' or \'female\'')
    bmi: float = Field(..., ge=10, le=60, description='body mass index (10-60)')
    children: int = Field(..., ge=0, le=10, description='number of children')
    smoker: str = Field(..., description='smoking status: \'yes\' or \'no\'')
    region: str = Field(..., description='residence region: \'northeast\', \'northwest\', \'southeast\', \'southwest\'')


    @validator('sex')
    def validate_sex(cls, v):
        if v.lower() not in ['male', 'female']:
            raise ValueError('sex must be \'male\' or \'female\'')
        return v.lower()

    @validator('smoker')
    def validate_smoker(cls, v):
        if v.lower() not in ['yes', 'no']:
            raise ValueError('smoker must be \'yes\' or \'no\'')
        return v.lower()

    @validator('region')
    def validate_region(cls, v):
        valid_regions = ['northeast', 'northwest', 'southeast', 'southwest']
        if v.lower() not in valid_regions:
            raise ValueError(f'region must be one of {valid_regions}')
        return v.lower()

    class Config:
        schema_extra = {
            'example' : {
                'age': 35,
                'sex': 'male',
                'bmi': 28.5,
                'children': 2,
                'smoker': 'no',
                'region': 'southeast'
            }
        }


class PredictionResponse(BaseModel):
    '''cost prediction response'''
    predicted_cost: float
    cost_tier: str
    clinical_note: str
    confidence_interval: list[float]
    recommendations: list[str]
    risk_factors: dict
    timestamp: str


class HealthResponse(BaseModel):
    '''health check response'''
    status: str
    version: str
    model_loaded: bool
    timestamp: str


# initialize get_predictor
predictor = None

@app.on_event('startup')
async def startup_event():
    '''load model on startup'''
    global predictor
    logger.info('loading cost prediction model...')
    try:
        predictor = get_predictor()
        logger.info('model loaded successfully')
    except Exception as e:
        logger.error(f'failed to load model: {e}')
        predictor = None



@app.get('/', tags=['root'])
async def root():
    '''root endpoint'''
    return {
        'message': 'healthcare cost prediction app',
        'docs': '/docs',
        'version': '1.0.0'
    }


@app.get('/health', response_model=HealthResponse, tags=['health'])
async def health_check():
    return HealthResponse(
        status='healthy' if predictor else 'unhealthy',
        version='1.0.0',
        model_loaded=predictor is not None,
        timestamp=datetime.now().isoformat()
    )



@app.post('/v1/predict', response_model=PredictionResponse, tags=['prediction'])
async def predict_cost(patient: PatientData):
    '''
    predict annual healtcare cost for a patient

    returns predicted cost, risk tier, and clinical recommendations
    '''
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='model not loaded, plese try again later'
        )

    logger.info (f'prediction request: age={patient.age}, smoker={patient.smoker}, region={patient.region}')

    try:
        result = predictor.predict(
            age=patient.age,
            sex=patient.sex,
            bmi=patient.bmi,
            children=patient.children,
            smoker=patient.smoker,
            region=patient.region
        )

        response = PredictionResponse(
            predicted_cost=result['predicted_cost'],
            cost_tier=result['cost_tier'],
            clinical_note=result['clinical_note'],
            confidence_interval=result['confidence_interval'],
            recommendations=result['recommendations'],
            risk_factors=result['risk_factors'],
            timestamp=datetime.now().isoformat()
        )

        logger.info(f'prediction successfull: ${result['predicted_cost']:.2f} - {result['cost_tier']} risk')

        return response

    except Exception as e:
        logger.error(f'predction error: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'prediction failed: {str(e)}'
        )


@app.post('/v1/batch-predict', tags=['prediction'])
async def batch_predict(patients: list[PatientData]):
    '''
    batch prediction for multiple patients
    '''
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='model not loaded'
        )

    results = []
    for patient in patients:
        try:
            result = predictor.predict(
                age=patient.age,
                sex=patient.sex,
                bmi=patient.bmi,
                children=patient.children,
                smoker=patient.smoker,
                region=patient.region
            )
            results.append({
                'patient': patient.dict(),
                'prediction':result
            })

        except Exception as e:
            logger.error(f'batch prediction error for {patient.dict(): {e}}')
            results.append({
                'patient': patient.dict(),
                'error': str(e)
            })

    return {
        'total patients': len(patients),
        'successful_predictions': len([r for r in results if 'error' not in r]),
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
