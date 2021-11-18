
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace mlnet.Services;

public class PredictionService
{
    private readonly ILogger<PredictionService> _logger;
    private static string ONNX_PATH => Path.Combine(Directory.GetCurrentDirectory(), "model.onnx");

    private MLContext mLContext;
    private ITransformer predictionPipeline;

    public PredictionService(ILogger<PredictionService> logger)
    {
        _logger = logger;
        this.mLContext = new MLContext();
        this.predictionPipeline = GetPredictionPipeline();
    }

    private ITransformer GetPredictionPipeline()
    {
        var inputColumns = new string[] { "dense_input" };
        var outputColumns = new string[] { "Identity" };

        var onnxPredictionPipeline = this.mLContext
                                    .Transforms
                                    .ApplyOnnxModel(
                                        outputColumnNames: outputColumns,
                                        inputColumnNames: inputColumns,
                                        ONNX_PATH);

        var emptyDv = this.mLContext.Data.LoadFromEnumerable(new OnnxInput[] { });
        return onnxPredictionPipeline.Fit(emptyDv);
    }

    public float Predict(float[] input)
    {
        this._logger.LogInformation("Starting prediction for {input}", input);
        var onnxPredictionEngine = this.mLContext.Model.CreatePredictionEngine<OnnxInput, OnnxOutput>(this.predictionPipeline);

        var prediction = onnxPredictionEngine.Predict(new OnnxInput { Input = input });
        this._logger.LogInformation("Predicted {output}", prediction.Output);

        return prediction.Output.First();
    }
}
