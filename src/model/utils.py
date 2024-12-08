def calculate_padding(kernel_size, dilation):
    return int(kernel_size * dilation - dilation / 2)