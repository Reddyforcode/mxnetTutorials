from mxnet import nd

# Custom array
arr = nd.array(((1, 2, 3, ), (5, 6, 7)))
print(arr)

# Full of ones
x = nd.ones((2, 3))
print(x)

#r random range and size of the vector
y = nd.random.uniform(-1, 1, (2, 3))
print(y)

#x  = nd.full((2, 3), 2.00)
#print(x)

values = (x.shape, x.size, x.dtype)
print(values)

##check if dimensions are correct
print(x*y)

print(x.exp())