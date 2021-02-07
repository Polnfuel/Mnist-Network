import numpy
import scipy.special
from PIL import Image

def reading_from_who_file_and_creating_numpy_array():
    with open("weights_hidden_output.txt", 'r') as weight_ho:
        weights_who = weight_ho.read()
    who_list = eval(weights_who)
    who_numpy_array = numpy.array(who_list)
    return who_numpy_array

def reading_from_wih_file_and_creating_numpy_array():
    with open("weights_input_hidden.txt", 'r') as weight_ih:
        weights_wih = weight_ih.read()
    wih_list = eval(weights_wih)
    wih_numpy_array = numpy.array(wih_list)
    return wih_numpy_array

def query(inputs_list, who_numpy_array, wih_numpy_array):
    inputs = numpy.array(inputs_list, ndmin=2).T

    hidden_inputs = numpy.dot(wih_numpy_array, inputs)
    hidden_outputs = activation_function(hidden_inputs)

    final_inputs = numpy.dot(who_numpy_array, hidden_outputs)
    final_outputs = activation_function(final_inputs)

    return final_outputs

if __name__ == '__main__':
    lstt = []
    img = numpy.array(Image.open("image.jpg"))
    for i in img:
        for j in i:
            if len(str(j)) == 7:
                lstt.append(255)
            if len(str(j)) == 13:
                lstt.append(0)

    who_numpy_array = reading_from_who_file_and_creating_numpy_array()
    wih_numpy_array = reading_from_wih_file_and_creating_numpy_array()

    activation_function = lambda x: scipy.special.expit(x)

    inputs = (numpy.asfarray(lstt) / 255.0 * 0.99) + 0.01
    outputs = query(inputs, who_numpy_array, wih_numpy_array)
    label = numpy.argmax(outputs)
    print(f"The Network label is {label}")
