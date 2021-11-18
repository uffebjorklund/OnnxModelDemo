using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;

namespace mlnet.Models;

public class OnnxInput
{
    [VectorType(1, 20)]
    [ColumnName("dense_input"), OnnxMapType(typeof(float), typeof(Single))]
    public float[] Input { get; set; }
}
