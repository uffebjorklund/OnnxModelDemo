# ONNX C# Test

A tiny proof of concept with `ML.NET`

## Idea

To load a trained `ONNX` model into `ML.NET` and then test the prediction engine by passing in each row of the `inferenceData.csv` file to the `Predition` endpoint.

## Instructions

### Build & Run

```
dotnet build
dotnet run
```

### Test

Take a line from the `inferenceData.csv` file and send as an array to the endpoint.
The model requires an array of floats with 20 items, more or less will cause an error

Example...
```
curl -X 'POST' \
  'https://localhost:5001/Prediction' \
  -H 'accept: text/plain' \
  -H 'Content-Type: application/json' \
  -d '[
  -0.011430789,-0.072804466,-0.072202496,0.010752341,10.005615,9.971158,10.071351,9.941953,-0.095031515,-0.082216434,-0.06416558,-0.060096428,10.077214,9.920494,9.9190235,-0.035336476,-0.08147322,0.043970365,-0.02370813,9.980851
]'
```




