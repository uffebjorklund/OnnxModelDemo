using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;

namespace mlnet.Models;

public class OnnxOutput
{
    [ColumnName("Identity"), OnnxMapType(typeof(float), typeof(Single))]
    public float[] Output { get; set; }
}
