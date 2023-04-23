from Bio import SeqIO
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import PySimpleGUI as gui
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

coords = []
seq = None
gb = None
plotReady = False
fasta_check = False
gb_check = False

# ---- MAIN WINDOW

layout = [
    [gui.Text("Программа предсказания ДНК-кодирующих участков генома")],
    [gui.Text("Выберите файлы соответствующих форматов")],
    [gui.Text('Выберите файл формата "fna" (fasta)'), gui.FileBrowse(key="file_fasta")],
    [
        gui.Text('Выберите файл формата "gbff" (genbank)'),
        gui.FileBrowse(key="file_genbank"),
    ],
    [
        #gui.Input("Trim start", size=(10), key="trim_start"),
        #gui.Input("Trim end", size=(10), key="trim_end"),
        gui.Input("Start", size=(10), key="start_coords"),
        gui.Input("End", size=(10), key="end_coords"),
        #gui.Input("Step", size=(10), key="step"),
        #gui.Input("Window", size=(10), key="win"),
    ],
    [
        gui.Button("Начать"),
        gui.Button("Показать график", key="showPlot")
    ],
    [gui.Output(size=(88, 20), key="output")],
]

window = gui.Window("Genemark algorythm", layout, finalize = True, size = (600, 400))

while True:
    
    event, values = window.read()
    
    if event == gui.WIN_CLOSED:
        break 

    if event == "Test":
        window["trim_start"].update(0)
        window["trim_end"].update(3000)
        window["start_coords"].update(2000)
        window["end_coords"].update(5996)
        window["step"].update(12)
        window["win"].update(96)      

    if event == "Начать":

        coords.clear()

        window["output"].Update("")

        if values["file_fasta"] != "" and values["file_genbank"] != "":
            if (
                values["start_coords"] != "Start"
                and values["end_coords"] != "End"
                #and values["step"] != "Step"
                #and values["win"] != "Window"
            ):

                with open(values["file_fasta"]) as file_fasta:
                    for record in SeqIO.parse(file_fasta, "fasta"):
                        seq = record.seq
                    fasta_check = True

                with open(values["file_genbank"]) as file_genbank:
                    for record in SeqIO.parse(file_genbank, "genbank"):
                        gb = record
                    gb_check = True

                if fasta_check == False or gb_check == False:
                    print("<!> Ошибка чтения файлов")

                else:
                    
                    coords = [int(values["start_coords"]), int(values["end_coords"])]
                    start, end = gb.features[0].location.start, None

                    ncod = []
                    cod = []

                    for i in range(len(gb.features)):
                        feature = gb.features[i]
                        fseq = feature.extract(seq)
                        end = feature.location.start

                        if (
                            feature.type == "CDS"
                            and fseq[:3] == "ATG"
                            and len(fseq) % 3 == 0
                        ):
                            cod.append(fseq.__str__())

                            if len(seq[start:end]) != 0:
                                ncod.append(seq[start:end].__str__())

                            start = feature.location.end

                    ## 2. Probabilities calculation
                    # 2.1 Initial Probabilities

                    def seq_probs(seq):
                        return np.array(
                            [
                                seq.count("T"),
                                seq.count("C"),
                                seq.count("A"),
                                seq.count("G"),
                            ]
                        ) / len(seq)

                    def cod_probs(seq):
                        res = []
                        for i in range(3):
                            res.append(seq_probs(seq[i::3]))
                        return np.array(res)

                    def make_table1(cod_seqs, ncod_seqs):
                        table1 = pd.DataFrame(
                            np.vstack(
                                (
                                    cod_probs("".join(cod_seqs)),
                                    seq_probs("".join(ncod_seqs)),
                                )
                            ).T,
                            index=["T", "C", "A", "G"],
                            columns=[f"pos{i}" for i in range(1, 4)] + ["nc"],
                        )
                        return table1

                    # 2.2 Transition probabilities

                    def cod_dprobs(seqs):
                        dcounts = dict(
                            zip(
                                [1, 2, 3],
                                [
                                    dict(
                                        zip(
                                            [
                                                "".join(pair)
                                                for pair in product("TCAG", repeat=2)
                                            ],
                                            [0] * 16,
                                        )
                                    )
                                    for i in range(3)
                                ],
                            )
                        )
                        for seq in seqs:
                            for i in range(1, len(seq)):
                                dcounts[i % 3 + 1][seq[i - 1 : i + 1]] += 1
                        return get_probs(dcounts)

                    def ncod_dprobs(seqs):
                        dcounts = dict(
                            zip(
                                ["".join(pair) for pair in product("TCAG", repeat=2)],
                                [0] * 16,
                            )
                        )
                        for seq in seqs:
                            for i in range(1, len(seq)):
                                dcounts[seq[i - 1 : i + 1]] += 1

                        return get_probs({0: dcounts})[0]

                    def get_probs(dcounts):
                        for pos in dcounts:
                            nuc_groups = dict(zip("TCAG", [0] * 4))
                            for dup in dcounts[pos]:
                                nuc_groups[dup[0]] += dcounts[pos][dup]
                            for dup in dcounts[pos]:
                                dcounts[pos][dup] /= nuc_groups[dup[0]]
                        return dcounts

                    def make_table2(cod_seqs, ncod_seqs):
                        table2 = pd.DataFrame(cod_dprobs(cod_seqs))
                        table2[4] = pd.Series(ncod_dprobs(ncod_seqs))
                        table2.rename(
                            columns=dict(
                                zip(
                                    np.arange(1, 5),
                                    [f"pos{i}" for i in range(1, 4)] + ["nc"],
                                )
                            ),
                            inplace=True,
                        )
                        table2.index = [
                            prob_notation(idx) for idx in table2.index.values
                        ]
                        return table2

                    def prob_notation(st):
                        return st[1] + "|" + st[0]

                    t1 = make_table1(cod, ncod)
                    print(t1)
                    t2 = make_table2(cod, ncod)
                    print()
                    print(t2)

                    ## 3. Predictions
                    # 3.1 Calculating predictions

                    def cod_proba(seq, t1, t2, frame=1):

                        if len(seq) == 0:
                            print("<!> Последовательность не получена")
                            return None

                        if frame not in [1, 2, 3]:
                            print("<!> Неверный порядок фрейма")
                            return None

                        prev_elem = seq[0]
                        start_pos = None

                        if frame == 1:
                            log_prob = np.log(t1["pos1"][seq[0]])
                            prev_pos = 1
                            for index, elem in enumerate(seq[1:]):
                                pair = "{}|{}".format(elem, prev_elem)
                                if prev_pos == 1:
                                    log_prob += np.log(t2["pos2"][pair])
                                    prev_pos = 2
                                elif prev_pos == 2:
                                    log_prob += np.log(t2["pos3"][pair])
                                    prev_pos = 3
                                else:
                                    log_prob += np.log(t2["pos1"][pair])
                                    prev_pos = 1
                                prev_elem = elem
                            prob = np.exp(log_prob)
                            return prob

                        elif frame == 2:
                            log_prob = np.log(t1["pos3"][seq[0]])
                            prev_pos = 3
                            for index, elem in enumerate(seq[1:]):
                                pair = "{}|{}".format(elem, prev_elem)
                                if prev_pos == 1:
                                    log_prob += np.log(t2["pos2"][pair])
                                    prev_pos = 2
                                elif prev_pos == 2:
                                    log_prob += np.log(t2["pos3"][pair])
                                    prev_pos = 3
                                else:
                                    log_prob += np.log(t2["pos1"][pair])
                                    prev_pos = 1
                                prev_elem = elem
                            prob = np.exp(log_prob)
                            return prob

                        elif frame == 3:
                            log_prob = np.log(t1["pos2"][seq[0]])
                            prev_pos = 2
                            for index, elem in enumerate(seq[1:]):
                                pair = "{}|{}".format(elem, prev_elem)
                                if prev_pos == 1:
                                    log_prob += np.log(t2["pos2"][pair])
                                    prev_pos = 2
                                elif prev_pos == 2:
                                    log_prob += np.log(t2["pos3"][pair])
                                    prev_pos = 3
                                else:
                                    log_prob += np.log(t2["pos1"][pair])
                                    prev_pos = 1
                                prev_elem = elem
                            prob = np.exp(log_prob)
                            return prob

                    def ncod_proba(seq, t1, t2):
                        if len(seq) == 0:
                            print("<!> Некодируемая последовательность не получена")
                            return None

                        log_prob_nc = np.log(t1["nc"][seq[0]])
                        prev_elem = seq[0]
                        for index, elem in enumerate(seq[1:]):
                            pair = "{}|{}".format(elem, prev_elem)
                            log_prob_nc += np.log(t2["nc"][pair])
                            prev_elem = elem
                        prob_nc = np.exp(log_prob_nc)
                        return prob_nc

                    def get_cod_probs(seq, t1, t2):
                        prob1 = cod_proba(seq, t1, t2, frame=1)
                        prob2 = cod_proba(seq, t1, t2, frame=2)
                        prob3 = cod_proba(seq, t1, t2, frame=3)
                        probnc = ncod_proba(seq, t1, t2)

                        res1 = (0.25 * prob1) / (
                            0.25 * prob1 + 0.25 * prob2 + 0.25 * prob3 + 0.25 * probnc
                        )
                        res2 = (0.25 * prob2) / (
                            0.25 * prob1 + 0.25 * prob2 + 0.25 * prob3 + 0.25 * probnc
                        )
                        res3 = (0.25 * prob3) / (
                            0.25 * prob1 + 0.25 * prob2 + 0.25 * prob3 + 0.25 * probnc
                        )
                        resnc = (0.25 * probnc) / (
                            0.25 * prob1 + 0.25 * prob2 + 0.25 * prob3 + 0.25 * probnc
                        )
                        return [res1, res2, res3, resnc]
                    
                    plotReady = True

            else:
                print("<!> Вы не ввели данные")

        else:
            print("<!> Вы не выбрали файлы")

    if event == "showPlot":
        
        if plotReady == True:
        
            def draw_figure(canvas, figure):
                figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
                figure_canvas_agg.draw()
                figure_canvas_agg.get_tk_widget().pack(side = 'top', fill = 'both', expand = 1)
                return figure_canvas_agg
            
            def open_plot_window():
                layout2 = [
                    [gui.Text('График')],
                    [gui.Canvas(key = "canvas")]
                ]
                window2 = gui.Window("Plot window", layout2, finalize = True, element_justification = 'center', size = (800, 900), location=(0,0))
                draw_figure(window2["canvas"].TKCanvas, plot_graph(pos_probs, start, end, step))
                
                while True:
                    event, values = window2.read()
                    if event == gui.WIN_CLOSED:
                        break                  
                        
                window2.close()
                
                
            sequence = seq[coords[0] : coords[1]]
            codon_list = ["ATG", "TAA", "TAG", "TGA"] #ATG - start, TAA, TAG, TGA - stop codons.

            found_codon_positions = []

            n = len(sequence)
            k = 0
            while k < n - 2:
                possible_codon = sequence[k : k + 3]
                if possible_codon in codon_list:
                    found_codon_positions.append(k)
                k += 1

            def plot_graph(data, start, end, step):
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 15))

                x = range(start, end, step)

                ax1.plot(x, data[:, 0])
                ax2.plot(x, data[:, 1])
                ax3.plot(x, data[:, 2])
                ax4.plot(x, data[:, 3])

                ax1.set_title("Вероятность кодирования при первом фрейме")
                ax2.set_title("Вероятность кодирования при втором фрейме")
                ax3.set_title("Вероятность кодирования при третьем фрейме")
                ax4.set_title('Вероятность некодирования')
                
                ax1.set_xlabel('Расстояние от начала отсчета', fontsize = 5)
                ax2.set_xlabel('Расстояние от начала отсчета', fontsize = 5)
                ax3.set_xlabel('Расстояние от начала отсчета', fontsize = 5)
                ax4.set_xlabel('Расстояние от начала отсчета', fontsize = 5)
                
                ax1.set_ylabel('Вероятность', fontsize = 5)
                ax2.set_ylabel('Вероятность', fontsize = 5)
                ax3.set_ylabel('Вероятность', fontsize = 5)
                ax4.set_ylabel('Вероятность', fontsize = 5)
                
                plt.subplots_adjust(wspace=0, hspace=1)
                
                return plt.gcf()

            pos_probs = list()
            start = 0
            end = 3000
            seq1 = seq[coords[0]:coords[1]]
            step = 12
            win = 96

            for i in range(start, end, step):
                pos_probs.append(get_cod_probs(seq1[i : i + win], t1, t2))
            pos_probs = np.array(pos_probs)
            
            open_plot_window()
        
        else:
            
            print("<!> Нечего показывать!")

window.close()
