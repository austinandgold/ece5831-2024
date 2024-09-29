
import logic_gate as lg

gate = lg.LogicGate()

#all input combinations
test_sets = [[0, 0],[1, 0],[0,1],[1,1]]


print("Testing AND gate")
for test in test_sets:

    y = gate.andgate(test[0],test[1])
    print(f'{y}=AND({test[0]}, {test[1]})')

print("Testing OR gate")
for test in test_sets:

    y = gate.orgate(test[0],test[1])
    print(f'{y}=OR({test[0]}, {test[1]})')

print("Testing NAND gate")
for test in test_sets:

    y = gate.nandgate(test[0],test[1])
    print(f'{y}=NAND({test[0]}, {test[1]})')

print("Testing NOR gate")
for test in test_sets:

    y = gate.norgate(test[0],test[1])
    print(f'{y}=NOR({test[0]}, {test[1]})')

print("Testing XOR gate")
for test in test_sets:

    y = gate.xorgate(test[0],test[1])
    print(f'{y}=XOR({test[0]}, {test[1]})')


