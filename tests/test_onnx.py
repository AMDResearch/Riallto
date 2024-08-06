#import torch
#import torch.nn as nn
#import torch.optim as optim
#import torchvision
#import torchvision.transforms as transforms
#from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat
#
#import os
#import pytest
#
#@pytest.mark.skip(reason="Skipping to ease load on CI.")
#def test_quantization():
#
#    torch.manual_seed(1337)
#
#    class MLP(nn.Module):
#        def __init__(self):
#            super(MLP, self).__init__()
#            self.fc1 = nn.Linear(28 * 28, 32)
#            self.relu = nn.ReLU()
#            self.fc2 = nn.Linear(32, 10)
#
#        def forward(self, x):
#            x = x.view(-1, 28 * 28)  # Flatten the input
#            x = self.fc1(x)
#            x = self.relu(x)
#            x = self.fc2(x)
#            return x
#
#    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
#
#    model = MLP()
#    criterion = nn.CrossEntropyLoss()
#    optimizer = optim.Adam(model.parameters(), lr=3e-4)
#
#    # Training loop
#    for epoch in range(5): 
#        running_loss = 0.0
#        for data in testloader:
#            inputs, labels = data
#            optimizer.zero_grad()
#
#            # Forward pass
#            outputs = model(inputs)
#            loss = criterion(outputs, labels)
#
#            # Backward pass
#            loss.backward()
#            optimizer.step()
#    print(f"Final loss: {loss.item()}")
#
#    try:
#        import vai_q_onnx
#    except:
#        raise ImportError("Failed to impot vai_q_onnx")
#
#    input_names = ['input']
#    output_names = ['output']
#    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
#
#    input_tensor = torch.randn(1,1,28,28)
#
#    torch.onnx.export(
#        model,
#        input_tensor,
#        "mnist.onnx",
#        export_params=True,
#        opset_version=13,
#        input_names=input_names,
#        output_names=output_names,
#        dynamic_axes=dynamic_axes,
#    )
#
#    class MNISTCalibrationDataReader(CalibrationDataReader):
#        def __init__(self, batch_size: int = 64):
#            super().__init__()
#            self.iterator = iter(testloader)
#
#        def get_next(self) -> dict:
#            try:
#                images, labels = next(self.iterator)
#                return {"input": images.numpy()}
#            except Exception:
#                return None
#
#    dr = MNISTCalibrationDataReader()
#
#    vai_q_onnx.quantize_static(
#            "mnist.onnx",
#            "mnist_quanized.onnx",
#            dr,
#            quant_format=QuantFormat.QDQ,
#            calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
#            activation_type=QuantType.QUInt8,
#            weight_type=QuantType.QInt8,
#            enable_dpu=True,
#            extra_options={'ActivationSymmetric': True} 
#        )
#
#    assert os.path.isfile("mnist_quanized.onnx")
