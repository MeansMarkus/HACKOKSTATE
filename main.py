def add(a, b):
    """Add two numbers together"""
    return a + b

def multiply(a, b):
    """Multiply two numbers"""
    return a * b



if __name__ == "__main__":
    x = 5
    y = 10
    print(f"Adding {x} and {y} gives {add(x, y)}")
    print(f"Multiplying {x} and {y} gives {multiply(x, y)}")