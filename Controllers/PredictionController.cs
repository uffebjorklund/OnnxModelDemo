using Microsoft.AspNetCore.Mvc;

namespace mlnet.Controllers;

[ApiController]
[Route("[controller]")]
public class PredictionController : ControllerBase
{
    private readonly PredictionService PredictionService;

    public PredictionController(PredictionService predictionService)
    {
        this.PredictionService = predictionService;
    }

    [HttpPost(Name = "Predict")]
    public float Predict(float[] input)
    {
        return this.PredictionService.Predict(input);
    }
}
