from npu.build import wslpath, is_win_path, is_wsl_win_path

def test_path_type_checks():
    assert(is_win_path("C:\\Users\\dev\\Riallto\\npu\\lib\\kernels\\cpp\\*.h"))
    assert(is_win_path("C:\\Users\\dev\\Riallto\\npu\\lib\\kernels\\cpp\\file.h"))
    assert(is_win_path("C:\\Users\\dev\\Riallto\\npu\\lib\\kernels\\cpp\\file.h"))
    assert(not is_win_path("/mnt/c/Users/dev/test.cpp"))
    assert(not is_win_path("/mnt/c/Users/dev/*.h"))
    assert(is_wsl_win_path("\\\\wsl.localhost\\Riallto\\dev\\Riallto\\npu\\lib\\kernels\\cpp\\file.h"))
    assert(is_wsl_win_path("\\\\wsl.localhost\\Riallto\\dev\\Riallto\\npu\\lib\\kernels\\cpp\\*.h"))
