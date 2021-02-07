import numpy
import scipy.special
from PIL import Image

class File_working():
    def __init__(self, numpy_arr_who, numpy_arr_wih):
        self.numpy_arr_who = numpy_arr_who
        self.numpy_arr_wih = numpy_arr_wih


    def writing_to_who_file(self):
        with open("weights_hidden_output.txt", 'w') as weights_ho:
            weights_ho.write("[")
            for i_who in self.numpy_arr_who:
                weights_ho.write("[")
                for j_who in i_who:
                    weights_ho.write(str(j_who))
                    if i_who[-1] == j_who:
                        continue
                    weights_ho.write(", ")
                if str(self.numpy_arr_who[-1]) == str(i_who):
                    weights_ho.write("]")
                    continue
                weights_ho.write("],\n")
            weights_ho.write("]")

    def reading_at_who_file_and_creating_numpy_array(self):
        with open("weights_hidden_output.txt", 'r') as weight_ho:
            weights_who = weight_ho.read()
        who_list = eval(weights_who)
        who_numpy_array = numpy.array(who_list)
        return who_numpy_array

    def writing_to_wih_file(self):
        with open("weights_input_hidden.txt", 'w') as weights_ih:
            weights_ih.write("[")
            for i_wih in self.numpy_arr_wih:
                weights_ih.write("[")
                for j_wih in i_wih:
                    weights_ih.write(str(j_wih))
                    if i_wih[-1] == j_wih:
                        continue
                    weights_ih.write(", ")
                if str(self.numpy_arr_wih[-1]) == str(i_wih):
                    weights_ih.write("]")
                    continue
                weights_ih.write("],\n")
            weights_ih.write("]")

    def reading_at_wih_file_and_creating_numpy_array(self):
        with open("weights_input_hidden.txt", 'r') as weight_ih:
            weights_wih = weight_ih.read()
        wih_list = eval(weights_wih)
        wih_numpy_array = numpy.array(wih_list)
        return wih_numpy_array

if __name__ == '__main__':
    lstt = []
    img = numpy.array(Image.open("image.jpg"))
    for i in img:
        for j in i:
            if len(str(j)) == 7:
                lstt.append(255)
            if len(str(j)) == 13:
                lstt.append(0)

    abc = File_working(0, 0)
    who_numpy_array = abc.reading_at_who_file_and_creating_numpy_array()
    wih_numpy_array = abc.reading_at_wih_file_and_creating_numpy_array()

    activation_function = lambda x: scipy.special.expit(x)

    def query(inputs_list, who_numpy_array, wih_numpy_array):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(wih_numpy_array, inputs)
        hidden_outputs = activation_function(hidden_inputs)

        final_inputs = numpy.dot(who_numpy_array, hidden_outputs)
        final_outputs = activation_function(final_inputs)

        return final_outputs

    inputs = (numpy.asfarray(lstt) / 255.0 * 0.99) + 0.01
    outputs = query(inputs, who_numpy_array, wih_numpy_array)
    label = numpy.argmax(outputs)
    print(f"The Network label is {label}")
