from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
app=FastAPI()
model=joblib.load("models/student_model.pkl")
@app.get("/")
def home():
    return {"message":"I HATE NISHADH"}
class Student_model(BaseModel):
    study_hours: float
    attendance: float
    previous_marks: float
@app.post("/predict")
def predict(student: Student_model):
    try:
        input_data=pd.DataFrame([{"study_hours":student.study_hours,"attendance":student.attendance,"previous_marks":student.previous_marks}])
        prediction=model.predict(input_data)
        return {"prediction":str(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

