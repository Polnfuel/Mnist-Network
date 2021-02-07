import numpy

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
