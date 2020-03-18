import numpy as np
import math
from funkcje_coco import fun_list, fun_name_list, fun_name_list_short
from algorithms import CDE, BBDE

if __name__ == "__main__":
    ################
    # Tests - CDE and BBDE
    ################
    print("###Tests - CDE and BBDE")
    file = open("project_report.txt", "w")
    file.write("###Tests - CDE and BBDE\n\n")
    #file with results
    tab_file = open("proj_table_results.txt", "w")
    #files with statistics
    tab_file_stat_CDE = open("proj_table_stat_CDE.txt", "w")
    tab_file_stat_BBDE = open("proj_table_stat_BBDE.txt", "w")
    tab_file_stat_CDE2 = open("proj_table_stat_CDE2.txt", "w")

    dimention = [2, 5, 8]
    vectors_number = [5, 7, 10]
    iteration_number = [50, 100, 500]
    #dimention = [2, 3]
    #vectors_number = [5, 6]
    #iteration_number = [20, 30]

    #repetit_num = 25
    repetit_num = 10
    result_tab_CDE = np.zeros([len(dimention), len(vectors_number), len(iteration_number), len(fun_list), repetit_num])
    result_tab_BBDE = np.zeros([len(dimention), len(vectors_number), len(iteration_number), len(fun_list), repetit_num])
    
    result_tab_CDE_mean = np.zeros([len(dimention), len(vectors_number), len(iteration_number), len(fun_list)])
    result_tab_BBDE_mean = np.zeros([len(dimention), len(vectors_number), len(iteration_number), len(fun_list)])
    result_tab_CDE_median = np.zeros([len(dimention), len(vectors_number), len(iteration_number), len(fun_list)])
    result_tab_BBDE_median = np.zeros([len(dimention), len(vectors_number), len(iteration_number), len(fun_list)])
    result_tab_CDE_stdv = np.zeros([len(dimention), len(vectors_number), len(iteration_number), len(fun_list)])
    result_tab_BBDE_stdv = np.zeros([len(dimention), len(vectors_number), len(iteration_number), len(fun_list)])
    result_tab_CDE_min = np.zeros([len(dimention), len(vectors_number), len(iteration_number), len(fun_list)])
    result_tab_BBDE_min = np.zeros([len(dimention), len(vectors_number), len(iteration_number), len(fun_list)])

    rep_i_num = 0
    dim_i_num = 0
    vec_i_num = 0
    iter_i_num = 0
    denominator = len(dimention)*len(vectors_number)*len(iteration_number)*len(fun_list)*repetit_num
    
    for dim_i_num in range(len(dimention)):
        dim = dimention[dim_i_num]
        print("###Testing for dimention = {}".format(dim))
        file.write("###Testing for dimention = {}\n".format(dim))
        bounds = [(-5, 5)] * dim
        for vec_i_num in range(len(vectors_number)):
            vec_num = vectors_number[vec_i_num]
            for iter_i_num in range(len(iteration_number)):
                iter_num = iteration_number[iter_i_num]
                for fun_iter in range(0, len(fun_list)):
                    for rep_i in range(repetit_num):
                        file.write("---Function: {}\n".format(fun_name_list[fun_iter]))
                        file.write("---Number of vectors: {}; Number of iterations: {}\n".format(vec_num, iter_num))
                        file.write("--CDE\n")
                        resultCDE = list(CDE(fun_list[fun_iter], bounds, vectors_num = vec_num, iterations=iter_num))
                        result_tab_CDE[dim_i_num][vec_i_num][iter_i_num][fun_iter][rep_i_num] = resultCDE[-1][1]
                        file.write("Best vector: " + np.array2string(resultCDE[-1][0]))
                        file.write("Fitness: " + str(resultCDE[-1][1]) + "\n")
                        file.write("--BBDE\n")
                        resultBBDE = list(BBDE(fun_list[fun_iter], bounds, vectors_num = vec_num, iterations=iter_num))
                        result_tab_BBDE[dim_i_num][vec_i_num][iter_i_num][fun_iter][rep_i_num] = resultBBDE[-1][1]
                        file.write("Best vector: " + np.array2string(resultBBDE[-1][0]))
                        file.write("Fitness: " + str(resultBBDE[-1][1]) + "\n")

                        #progress = (fun_iter*iter_i_num*vec_i_num*dim_i_num*rep_i*100)/denominator
                        #b1 = "Progress = {}".format(progress)
                        #print(b1, end="\r")
                    result_tab_CDE_mean[dim_i_num][vec_i_num][iter_i_num][fun_iter] = np.mean(result_tab_CDE[dim_i_num][vec_i_num][iter_i_num][fun_iter])
                    result_tab_BBDE_mean[dim_i_num][vec_i_num][iter_i_num][fun_iter] = np.mean(result_tab_BBDE[dim_i_num][vec_i_num][iter_i_num][fun_iter])
                    result_tab_CDE_median[dim_i_num][vec_i_num][iter_i_num][fun_iter] = np.median(result_tab_CDE[dim_i_num][vec_i_num][iter_i_num][fun_iter])
                    result_tab_BBDE_median[dim_i_num][vec_i_num][iter_i_num][fun_iter] = np.median(result_tab_BBDE[dim_i_num][vec_i_num][iter_i_num][fun_iter])
                    result_tab_CDE_stdv[dim_i_num][vec_i_num][iter_i_num][fun_iter] = np.std(result_tab_CDE[dim_i_num][vec_i_num][iter_i_num][fun_iter])
                    result_tab_BBDE_stdv[dim_i_num][vec_i_num][iter_i_num][fun_iter] = np.std(result_tab_BBDE[dim_i_num][vec_i_num][iter_i_num][fun_iter])
                    result_tab_CDE_min[dim_i_num][vec_i_num][iter_i_num][fun_iter] = np.amin(result_tab_CDE[dim_i_num][vec_i_num][iter_i_num][fun_iter])
                    result_tab_BBDE_min[dim_i_num][vec_i_num][iter_i_num][fun_iter] = np.amin(result_tab_BBDE[dim_i_num][vec_i_num][iter_i_num][fun_iter])

    ################
    # Tests - CDE(F, CR)
    ################
    print("###Tests - CDE(F, CR)")
    file.write("\n\n###Tests - CDE(F, CR)\n\n")
    file.write("---Number of vectors: 5; Number of iterations: 1000; Dimention: 5\n".format(vec_num, iter_num))

    F_values = [0, 0.4, 0.8, 1]
    CR_values = [0, 0.3, 0.7, 1]
    bounds = [(-5, 5)] * 5
    result_tab_CDE2 = np.zeros([len(F_values), len(CR_values), len(fun_list), repetit_num])

    result_tab_CDE2_mean = np.zeros([len(F_values), len(CR_values), len(fun_list)])
    result_tab_CDE2_median = np.zeros([len(F_values), len(CR_values), len(fun_list)])
    result_tab_CDE2_stdv = np.zeros([len(F_values), len(CR_values), len(fun_list)])
    result_tab_CDE2_min = np.zeros([len(F_values), len(CR_values), len(fun_list)])

    F_i_num = 0
    CR_i_num = 0
    rep_i_num = 0
    denominator = len(F_values)*len(CR_values)*len(fun_list)*repetit_num
    
    for F_i_num in range(len(F_values)):
        F_val = F_values[F_i_num]
        for CR_i_num in range(len(CR_values)):
            CR_val = CR_values[CR_i_num]
            for fun_iter in range(0, len(fun_list)):
                for rep_i in range(repetit_num):
                    file.write("---Function: {}\n".format(fun_name_list[fun_iter]))
                    #resultCDE = list(CDE(fun_list[fun_iter], bounds, F = F_val, CR = CR_val))
                    resultCDE = list(CDE(fun_list[fun_iter], bounds, F = F_val, CR = CR_val, iterations=500))
                    result_tab_CDE2[F_i_num][CR_i_num][fun_iter][rep_i] = resultCDE[-1][1]
                    file.write("Best vector: " + np.array2string(resultCDE[-1][0]) + "\n")
                    file.write("Fitness: " + str(resultCDE[-1][1]) + "\n")

                    #progress = (fun_iter*CR_i_num*F_i_num*rep_i*100)/denominator
                    #b1 = "Progress = {}".format(progress)
                    #print(b1, end="\r")

                result_tab_CDE2_mean[F_i_num][CR_i_num][fun_iter] = np.mean(result_tab_CDE2[F_i_num][CR_i_num][fun_iter])
                result_tab_CDE2_median[F_i_num][CR_i_num][fun_iter] = np.median(result_tab_CDE2[F_i_num][CR_i_num][fun_iter])
                result_tab_CDE2_stdv[F_i_num][CR_i_num][fun_iter] = np.std(result_tab_CDE2[F_i_num][CR_i_num][fun_iter])
                result_tab_CDE2_min[F_i_num][CR_i_num][fun_iter] = np.amin(result_tab_CDE2[F_i_num][CR_i_num][fun_iter])

    ################
    # Saving tables to the files
    ################

    print("Saving tables to a file...")

    tab_file_stat_CDE.write("\n\nTests - CDE\n\n")
    temp1 = np.array(["iter."])
    row_titl = np.append(temp1, iteration_number)[:, np.newaxis]
    col_titl = np.asarray(fun_name_list_short)[np.newaxis, :]
    for dim_i_num in range(len(dimention)):
        for vec_i_num in range(len(vectors_number)):
            tab_file_stat_CDE.write("\ndimention = {}    vectors = {}\n\n".format(dimention[dim_i_num], vectors_number[vec_i_num]))
            tab_file_stat_CDE.write("Mean value:\n")
            temp2 = np.vstack((col_titl, result_tab_CDE_mean[dim_i_num][vec_i_num]))
            np.savetxt(tab_file_stat_CDE, np.hstack((row_titl, temp2)),delimiter='\t', fmt='%s')
            tab_file_stat_CDE.write("Median:\n")
            temp2 = np.vstack((col_titl, result_tab_CDE_median[dim_i_num][vec_i_num]))
            np.savetxt(tab_file_stat_CDE, np.hstack((row_titl, temp2)),delimiter='\t', fmt='%s')
            tab_file_stat_CDE.write("Standard deviation:\n")
            temp2 = np.vstack((col_titl, result_tab_CDE_stdv[dim_i_num][vec_i_num]))
            np.savetxt(tab_file_stat_CDE, np.hstack((row_titl, temp2)),delimiter='\t', fmt='%s')
            tab_file_stat_CDE.write("Minimum:\n")
            temp2 = np.vstack((col_titl, result_tab_CDE_min[dim_i_num][vec_i_num]))
            np.savetxt(tab_file_stat_CDE, np.hstack((row_titl, temp2)),delimiter='\t', fmt='%s')

    tab_file_stat_BBDE.write("\n\nTests - BBDE\n\n")
    for dim_i_num in range(len(dimention)):
        for vec_i_num in range(len(vectors_number)):
            tab_file_stat_BBDE.write("\ndimention = {}    vectors = {}\n\n".format(dimention[dim_i_num], vectors_number[vec_i_num]))
            tab_file_stat_BBDE.write("Mean value:\n")
            temp2 = np.vstack((col_titl, result_tab_BBDE_mean[dim_i_num][vec_i_num]))
            np.savetxt(tab_file_stat_BBDE, np.hstack((row_titl, temp2)),delimiter='\t', fmt='%s')
            tab_file_stat_BBDE.write("Median:\n")
            temp2 = np.vstack((col_titl, result_tab_BBDE_median[dim_i_num][vec_i_num]))
            np.savetxt(tab_file_stat_BBDE, np.hstack((row_titl, temp2)),delimiter='\t', fmt='%s')
            tab_file_stat_BBDE.write("Standard deviation:\n")
            temp2 = np.vstack((col_titl, result_tab_BBDE_stdv[dim_i_num][vec_i_num]))
            np.savetxt(tab_file_stat_BBDE, np.hstack((row_titl, temp2)),delimiter='\t', fmt='%s')
            tab_file_stat_BBDE.write("Minimum:\n")
            temp2 = np.vstack((col_titl, result_tab_BBDE_min[dim_i_num][vec_i_num]))
            np.savetxt(tab_file_stat_BBDE, np.hstack((row_titl, temp2)),delimiter='\t', fmt='%s')

    tab_file_stat_CDE2.write("\n\nTests - CDE(F, CR)\n\n")
    temp1 = np.array(["CR"])
    row_titl = np.append(temp1, CR_values)[:, np.newaxis]
    col_titl = np.asarray(fun_name_list_short)[np.newaxis, :]
    for F_i_num in range(len(F_values)):
        tab_file_stat_CDE2.write("\nF = {} \n\n".format(F_values[F_i_num]))
        tab_file_stat_CDE2.write("Mean value:\n")
        temp2 = np.vstack((col_titl, result_tab_CDE2_mean[F_i_num]))
        np.savetxt(tab_file_stat_CDE2, np.hstack((row_titl, temp2)),delimiter='\t', fmt='%s')
        tab_file_stat_CDE2.write("Median:\n")
        temp2 = np.vstack((col_titl, result_tab_CDE2_median[F_i_num]))
        np.savetxt(tab_file_stat_CDE2, np.hstack((row_titl, temp2)),delimiter='\t', fmt='%s')
        tab_file_stat_CDE2.write("Standard deviation:\n")
        temp2 = np.vstack((col_titl, result_tab_CDE2_stdv[F_i_num]))
        np.savetxt(tab_file_stat_CDE2, np.hstack((row_titl, temp2)),delimiter='\t', fmt='%s')
        tab_file_stat_CDE2.write("Minimum:\n")
        temp2 = np.vstack((col_titl, result_tab_CDE2_min[F_i_num]))
        np.savetxt(tab_file_stat_CDE2, np.hstack((row_titl, temp2)),delimiter='\t', fmt='%s')
    
    row_titl = np.asarray(fun_name_list_short)[:, np.newaxis]
    
    tab_file.write("\n\nPURE RESULTS\n\n")

    tab_file.write("Tests - CDE\n\n")
    for dim_i_num in range(len(dimention)):
        for vec_i_num in range(len(vectors_number)):
            for iter_i_num in range(len(iteration_number)):
                tab_file.write("\ndimentions = {}    vectors = {}    iterations = {}\n\n".format(dimention[dim_i_num], vectors_number[vec_i_num], iteration_number[iter_i_num]))
                np.savetxt(tab_file, np.hstack((row_titl, result_tab_CDE[dim_i_num][vec_i_num][iter_i_num])),delimiter='\t', fmt='%s')

    tab_file.write("\n\nTests - BBDE\n\n")
    for dim_i_num in range(len(dimention)):
        for vec_i_num in range(len(vectors_number)):
            for iter_i_num in range(len(iteration_number)):
                tab_file.write("\ndimentions = {}    vectors = {}    iterations = {}\n\n".format(dimention[dim_i_num], vectors_number[vec_i_num], iteration_number[iter_i_num]))
                np.savetxt(tab_file, np.hstack((row_titl, result_tab_BBDE[dim_i_num][vec_i_num][iter_i_num])),delimiter='\t', fmt='%s')
    
    tab_file.write("\n\nTests - CDE(F, CR)\n\n")
    for F_i_num in range(len(F_values)):
        for CR_i_num in range(len(CR_values)):
            tab_file.write("\nF = {}  CR = {}\n\n".format(F_values[F_i_num], CR_values[CR_i_num]))
            np.savetxt(tab_file, np.hstack((row_titl, result_tab_CDE2[F_i_num][CR_i_num])),delimiter='\t', fmt='%s')
    
    file.close()
    tab_file.close()
    tab_file_stat_CDE.close()
    tab_file_stat_BBDE.close()
    tab_file_stat_CDE2.close()
    print("Report generation completed. Check: project_report.txt and proj_table*.txt")
