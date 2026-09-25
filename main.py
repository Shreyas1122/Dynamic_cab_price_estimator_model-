from fastapi import FastAPI
from pydantic import BaseModel,Field
import joblib
from typing import Literal
import pandas as pd 
from fastapi.middleware.cors import CORSMiddleware

model =joblib.load("Dynamic_pricing.pkl")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#Creating a validation class for the input variables 
class Inputvariables (BaseModel):
    Number_of_Riders:int=Field(...,ge=0,le=100000)
    Number_of_Drivers:int=Field(...,ge=0,le=10000)
    Number_of_Past_Rides:int=Field(...,ge=0,le=100000)
    Time_of_Booking:Literal["Morning","Afternoon","Evening","Night"]
    Vehicle_Type:Literal["Premium","Economy"]  
    Expected_Ride_Duration:int =Field(...,ge=0,le=300)


#creating class for the data output 
class Outputprice(BaseModel):
    Estimated_price:float





@app.get("/")
def fun():
    return "Hello world"



@app.post("/predict")
def prediction(data:Inputvariables):
    dataframe=pd.DataFrame(data={
        "Number_of_Riders":data.Number_of_Riders,
        "Number_of_Drivers":data.Number_of_Drivers,
        "Number_of_Past_Rides":data.Number_of_Past_Rides,
        "Time_of_Booking":data.Time_of_Booking,
        "Vehicle_Type":data.Vehicle_Type,
        "Expected_Ride_Duration":data.Expected_Ride_Duration
    },index=[0])

    prediction_data=model.predict(dataframe)
    return Outputprice(Estimated_price=round(float(prediction_data),2))


    








