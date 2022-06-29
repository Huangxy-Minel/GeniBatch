import numpy as np
from federatedml.FATE_Engine.python.BatchPlan.planner.batch_plan import BatchPlan
from federatedml.FATE_Engine.python.BatchPlan.storage.data_store import DataStorage
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder
from federatedml.FATE_Engine.python.BatchPlan.encryption.encrypt import BatchEncryption
from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierPublicKey, PaillierPrivateKey

from federatedml.secureprotol import PaillierEncrypt
from federatedml.util.fixpoint_solver import FixedPointEncoder

def encrypt_decrypt():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=64)
    matrixA = np.random.uniform(-1, 1, (1, 100))     # ciphertext
    matrixB = np.random.uniform(-1, 1, (100, 1))     # plaintext

    '''Contruct BatchPlan'''
    myBatchPlan.fromMatrix(matrixA, True)
    myBatchPlan.matrixMul([matrixB])
    print("\n-------------------Batch Plan before weave:-------------------")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Batch Plan after weave:-------------------")
    myBatchPlan.weave()
    batch_scheme = myBatchPlan.getBatchScheme()
    max_element_num, split_num = batch_scheme[0]
    print("Element num in one vector: ", + max_element_num)
    print("Split num: ", + split_num)

    '''Encrypt'''
    print("\n-------------------Encryption:-------------------")
    print("Plaintext: ")
    print(matrixA)
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()
    encrypted_row_vec = myBatchPlan.encrypt(matrixA, batch_scheme[0], encrypter.public_key)
    '''Decrypt'''
    decrypted_vec = myBatchPlan.decrypt(encrypted_row_vec, encrypter.privacy_key)
    print("-------------------After decryption:-------------------")
    print(decrypted_vec)

def encrypt_decrypt_with_gpu_encode():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=64, device_type='GPU')
    matrixA = np.random.uniform(-1, 1, (1, 100))     # ciphertext
    matrixB = np.random.uniform(-1, 1, (100, 1))     # plaintext

    '''Contruct BatchPlan'''
    myBatchPlan.fromMatrix(matrixA, True)
    myBatchPlan.matrixMul([matrixB])
    print("\n-------------------Batch Plan before weave:-------------------")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Batch Plan after weave:-------------------")
    myBatchPlan.weave()
    batch_scheme = myBatchPlan.getBatchScheme()
    max_element_num, split_num = batch_scheme[0]
    print("Element num in one vector: ", + max_element_num)
    print("Split num: ", + split_num)

    '''Encrypt'''
    print("\n-------------------Encryption:-------------------")
    print("Plaintext: ")
    print(matrixA)
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()
    encrypted_row_vec = myBatchPlan.encrypt(matrixA, batch_scheme[0], encrypter.public_key)
    '''Decrypt'''
    decrypted_vec = myBatchPlan.decrypt(encrypted_row_vec, encrypter.privacy_key)
    print("-------------------After decryption:-------------------")
    print(decrypted_vec)

def encrypted_add():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=64, device_type='GPU')
    matrixA = np.random.uniform(-1, 1, (1, 100))     # ciphertext
    matrixB = np.random.uniform(-1, 1, (1, 100))     # plaintext

    '''Contruct BatchPlan'''
    myBatchPlan.fromMatrix(matrixA, True)
    myBatchPlan.matrixAdd([matrixB], [False])
    print("\n-------------------Batch Plan before weave:-------------------")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Batch Plan after weave:-------------------")
    myBatchPlan.weave()
    batch_scheme = myBatchPlan.getBatchScheme()
    max_element_num, split_num = batch_scheme[0]
    print("Element num in one vector: ", + max_element_num)
    print("Split num: ", + split_num)

    '''Encrypt'''
    print("\n-------------------Encryption:-------------------")
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()
    encrypted_row_vec = myBatchPlan.encrypt(matrixA, batch_scheme[0], encrypter.public_key)

    '''Assign encrypted vector'''
    myBatchPlan.assignEncryptedVector(0, 0, encrypted_row_vec)

    print("\n-------------------Begin to exec Batch Plan.-------------------")
    outputs = myBatchPlan.parallelExec()
    '''Decrypt'''
    outputs = [myBatchPlan.decrypt(output, encrypter.privacy_key) for output in outputs]
    print(outputs)
    row_num, col_num = myBatchPlan.matrix_shape
    output_matrix = np.zeros(myBatchPlan.matrix_shape)
    for row_id in range(row_num):
        output_matrix[row_id, :] = outputs[row_id][0:col_num]
    print("\n-------------------Batch Plan output:-------------------")
    print(output_matrix)
    print("\n-------------------Numpy output:-------------------")
    result = matrixA + matrixB
    print(result)
    if np.allclose(output_matrix, result):
        print("\n-------------------Test Pass!-------------------")
    else:
        print("\n-------------------Test Fail-------------------")
        print(output_matrix == result)

def encrypted_mul():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=64, device_type='GPU')
    # matrixA = np.random.uniform(-1, 1, (1, 300))     # ciphertext
    # matrixB = np.random.uniform(-1, 1, (1, 300))
    # matrixC = np.random.uniform(-1, 1, (300, 1))     # plaintext

    matrixA = [0.9141486905086378, -0.11433326872993721, -0.9941312543438079, -0.23007433671683652, 0.3566968447914214, 0.9567841357611491, 0.8057728440953424, 0.720418051438662, -0.3318768347090941, 0.37765506228866697, -0.12600548708014703, -0.7167344175182198, 0.876088245064327, 0.8563716406598363, -0.3360126071222964, -0.8059281922672845, -0.811804381429396, 0.49127592145328314, -0.5221106618695557, -0.8256284320527114, -0.6086556026920715, -0.39330690030652504, 0.349251160749827, -0.12693580745815192, 0.9800915257267078, 0.141449709871035, -0.0455563216260928, -0.4829959090101874, -0.9493953163478763, -0.5868562158442328, -0.656122097124904, -0.9166255261643281, 0.03769526296312531, 0.16295946713239906, 0.9485941969090792, 0.6227607233166408, 0.785319854027209, -0.8217031439801352, -0.7806124026650536, 0.26896687424078003, -0.46968917112498754, 0.820660593861207, 0.42672470921236383, 0.5550514656226349, -0.6266315520625529, 0.04186422235761822, 0.682103300744918, 0.3600987031370757, 0.08176426860862596, 0.4814803316728262, 0.1853004206170632, 0.6057344666534601, -0.8908102184910642, -0.004905334124096772, -0.5277196466801841, -0.007072069545816406, -0.19313441920880958, -0.6919212119070588, 0.46348465964822716, -0.5299291563143922, 0.12001553242743901, -0.9248102537340734, 0.9890366178875221, 0.9534946563595563, 0.2229600318323659, -0.7150559048533838, 0.579078668198167, -0.9162062709561287, 0.6516817280588196, -0.9151700429635752, -0.11008792879930751, -0.5890630233619614, 0.3754691083057018, -0.878517307159471, 0.08285099342662083, -0.109390692046621, 0.012199936786365662, 0.7391492641560746, -0.32198842975247755, 0.7522596555057095, -0.718233336714275, 0.6304952861220627, -0.41089642179755814, 0.6541957558479614, 0.12967990107280292, -0.12387744741605866, -0.715392837855805, -0.7425646662503633, -0.2774918988407715, -0.17392057269243422, -0.5371139567846783, 0.7609633837467142, 0.33724463534057825, 0.038943670529041574, -0.17017280059400508, -0.16865267599072675, -0.5693080676294624, -0.18824924031391976, -0.013378259748556776, -0.42903708355544046, 0.9311314666980575, -0.8558968330026948, 0.09644723414682899, -0.697224819533663, -0.2566337154453824, -0.030974792433530185, 0.817563472471627, 0.3315680390660407, -0.3894054087615828, -0.4349259379498631, 0.6806743655492122, -0.056302228284652145, -0.3301042975699269, 0.4316994048300329, -0.41216294901465766, 0.548077217414404, -0.5150172970229783, 0.8306130505846223, 0.6818766131378957, 0.33150677428688735, -0.6151487883647326, 0.040727657516540505, 0.5338407464181183, 0.6823032215361571, -0.26242135990678417, 0.9204190310293832, 0.7486274900479528, 0.7122249547075579, 0.7524362237839555, -0.45090913763051765, 0.9739775836661888, 0.699952075320228, 0.03596555103078436, -0.035392195921617686, -0.8715336118019896, 0.6891375077646773, 0.3363887447662963, -0.0297466720517936, -0.9219126715202663, -0.7311219366062869, 0.8302252362311424, 0.8094635911573289, -0.8091282577954015, -0.4200364665732701, -0.3558754707400036, -0.8885919939534324, 0.47241693436834087, 0.07169775631069975, -0.09082161409067346, -0.08128800346874421, 0.44533720513239894, -0.5637802584374807, -0.9793583222702074, 0.29481527416841113, -0.07156915169114186, 0.1876243039388219, -0.7653045110657362, 0.2562430936256517, -0.11654818237126174, -0.019066753261033798, -0.3755810836543534, -0.642691786774185, 0.9041909555997736, -0.4499988887459301, 0.11475136878940306, -0.5105302921373616, 0.4548330678064354, 0.3376603903282538, -0.9702869568438559, -0.10769614883115941, 0.40096651685113205, 0.33205834039585036, -0.5721882613075591, -0.3428628969618195, 0.8220768999667623, 0.46761306413107095, 0.552536072827341, 0.7678738994474945, 0.4965923713012175, -0.3285005396962377, 0.9947746497978631, 0.13036031294393124, 0.12051633983279109, -0.09461068199499834, 0.22730303149334707, 0.7686482367818526, 0.1478700868542635, -0.9569012588021317, 0.5235760465891384, 0.2026644717584214, 0.5340308414080619, -0.20451278587179833, -0.34168800157344204, 0.6439465978680123, 0.1157637591932279, 0.5832880249243866, -0.45819849589745076, 0.1651972935995114, 0.1462207046842643, 0.5375338822686966, 0.9282199397324873, 0.9217754554252391, -0.044363543747128364, 0.499868550029021, 0.13748880586933332, 0.16432612770729782, 0.6659850624108388, -0.19747414598134583, 0.5804929517134783, 0.6285665259585631, -0.9336309388899375, 0.8705682680861544, -0.7312079200660191, 0.3456314273394072, 0.944980209654853, -0.5976908478589928, 0.6052822579514041, 0.0074159887499198884, 0.37641529762827686, -0.9093734059908685, 0.367202461032903, 0.36153855112536104, -0.901527955726944, -0.4201723733289353, -0.22097841155337306, -0.49982493213603885, 0.26724891410425755, -0.05439515173146603, 0.65941772190265, 0.31559662033402036, -0.27557652091400686, 0.10691774088540185, -0.5308588070559066, 0.25544160546767514, -0.3161841191126311, 0.21719888151566535, 0.7066844000453363, -0.9738686848101126, 0.6338509224027751, -0.13856178901706717, 0.14758374712995526, 0.11972068960186477, -0.8195255126051424, 0.7133073013554845, -0.6503184617311926, 0.6916209685897994, 0.886270154139779, -0.8192785158502498, -0.9660134510390161, 0.8626594163858587, -0.9323045485578292, 0.20810270688212507, -0.06742244363606953, 0.1549615003439837, -0.03373177861867682, -0.6421660522155717, 0.5561244028636518, -0.18302446655036708, -0.004719843926916489, -0.9538481391875682, 0.8089362574701591, 0.9163077918384641, 0.17946880130610254, -0.9887791086702975, 0.8704416377027961, -0.19122984822879618, 0.6600986915636844, 0.5363906512908081, -0.7941234315012096, 0.8971244027439591, 0.17785872889962073, -0.23460613271636221, 0.45323068044929093, -0.37916901014751425, -0.2292039779224233, -0.4726081087763223, -0.2819808786167515, -0.5390087749561121, -0.8765553008869422, -0.7417234018137222, -0.02616209022087168, 0.9526432329767738, 0.21914592599760518, -0.8633919367552381, 0.8973258853182102, -0.5086836570800404, -0.2707627779428794, -0.2381225894930208, 0.004676109342082402, 0.8466636301697013, -0.40847853749994445, 0.41135313355399683, -0.2241596150270193, 0.5710095968489637, -0.8276607097758255, 0.8401368787007146, -0.4866979261100488, -0.8045325709772246, 0.43037477525664825, 0.5276523724479132]
    matrixC = [0.2626991836976007, 0.4422178928473024, 0.40600612421015847, -0.9962131291879412, 0.21635087163270716, 0.8270048389423357, 0.4966537577807044, -0.43601414006996175, 0.6331135403126313, 0.13037567060739708, 0.002357612787750485, -0.029959220317332225, -0.4118127421572264, 0.23083072326181897, -0.030469425901066494, -0.04300964153104547, 0.98125121337917, -0.3752475377530331, -0.3237278352141577, -0.4348046006949351, -0.5568444269875712, -0.3011516153327154, 0.8532599318661953, -0.450052005605037, -0.14435853327448767, 0.32527250918608086, 0.017340830982769617, 0.012899294213478463, -0.658685658124877, 0.14618076111359035, 0.9610384877750335, -0.37945660430830497, -0.5429092184339701, -0.3839950371796712, 0.57586062923688, 0.38501390473417874, 0.5692730915562823, 0.5829903998561581, 0.326288237555056, 0.3849074552028555, -0.052940314561903845, -0.7713684485647727, -0.36898708029990623, -0.6001239558793459, 0.48872127278758914, -0.5042015743175217, -0.02545882532887389, 0.9430595289150245, 0.49098939023184385, 0.4319162172786004, 0.1873866342994328, -0.31993617568652377, -0.6679721500446452, 0.8478382653990879, 0.4392584853318313, 0.0952595752651404, 0.4427580184699669, 0.4917617340059932, 0.7425052560555563, 0.10847496491600395, 0.23822319017575588, 0.4614071746570729, -0.2832235007837709, -0.36457100927815533, -0.914435353528118, 0.4857503465385189, 0.4738653262794743, 0.09790956394596373, 0.5565596545348481, 0.28421049394113673, -0.7407596563345975, 0.05207844232547787, -0.41923206294191906, 0.8694339222579046, 0.026695829201547783, 0.8536380064994351, -0.9580019981209549, 0.9367843204005417, -0.5008434725702955, -0.4524410121431046, 0.3416376422563714, 0.9584223229803308, 0.09594991103662509, -0.19060024492514582, -0.5124511771581961, 0.3084392935157134, -0.48052986657554264, -0.743233457799245, 0.8472838073656979, 0.4650510625908939, -0.6814718684222698, -0.8509774914927801, 0.02503229916773586, 0.14216213210745376, -0.8523159124236594, 0.8961670061561751, 0.6250623306873391, -0.4278308615533215, -0.06397738571428557, -0.5512751566590737, -0.06338184062221086, 0.20317515425661425, -0.38691609742951605, 0.287621917729872, -0.9680648885697218, 0.8355252541054068, -0.7945355685739173, -0.08703042764559621, -0.9201204675485206, 0.8655051926075794, 0.16533115157769518, -0.21355963915595133, 0.7991087033627668, -0.49107166291987303, -0.7647668217469936, 0.4430271591050787, -0.7666847596874333, 0.5319467759698135, 0.25274897406205477, -0.44396704408853904, -0.896865911266228, 0.2569848510958921, 0.1982558819377871, 0.5663603176261038, -0.4937944228575215, 0.4891707961355394, 0.3499328965146007, -0.9182995585782654, -0.39915656559198465, 0.5761524815878676, -0.17031168435787047, 0.28864556389515905, 0.921205236458601, 0.6474213492587233, 0.06586749596502428, 0.0445222921733166, 0.6264102672296374, 0.27114403232133655, 0.4125823439483127, 0.8564081938612365, -0.584834412961303, 0.8837100169461316, 0.2809651831124591, 0.3019401510959163, 0.9071226971963329, 0.7899190104519771, 0.4812125352008647, 0.977478031957798, 0.30709531286342706, 0.9651543325677261, 0.48453636542844736, 0.11650345438935839, 0.0484495472640194, 0.36137338916246486, -0.16638138971631333, -0.9252636289629674, -0.34662268585774125, -0.20992489501778477, -0.8958234846559114, -0.5866983051907226, 0.6470090763997953, -0.4227754058012805, -0.3304750705626853, 0.08910327459396217, 0.8339798181211844, -0.35762115307343856, 0.5057321695359285, 0.24900017933347574, -0.5720042110796979, 0.02864891129659486, 0.7572563701562816, 0.6257297834054623, 0.04736512816159699, -0.760154217736986, -0.40244883825325206, -0.40909413104330405, 0.6411311099958967, -0.5342242346589081, -0.9256570311595866, -0.5132327074804668, 0.622237656870926, 0.7486276714865769, 0.715178252520847, 0.7200302922821287, -0.4907295448868554, -0.19154670500273152, 0.7428856782241879, 0.6080987798310411, -0.5057931422034065, -0.0166068718612169, -0.10735441721925087, 0.6337394939832541, 0.13884548143435982, -0.42887937142975163, -0.9662739745857016, 0.26284124803063547, 0.8689865888036876, 0.6896234214143162, 0.16456710977166145, 0.24648201012868087, -0.500027918854091, -0.1166655944610675, -0.18880014506199938, 0.035414995536799276, -0.08837012866427729, -0.8975796721515834, -0.8493783474069958, -0.6466833498888813, 0.5342816146390412, 0.9135649346474539, 0.010581726073866315, 0.2085551895382196, -0.059041243262089704, -0.7650741361746152, 0.14938651982157958, -0.9021312671675452, 0.8289049004895934, -0.9867789531985047, -0.6426828471570378, -0.17361176223467112, -0.5778311860769305, 0.6304326550118247, 0.30374847375671843, 0.9040440493288135, -0.1220061482370558, 0.2650364189213099, -0.7408167510957835, 0.4135033226632794, -0.34545617465835243, -0.8052107867528486, -0.9950459059711252, -0.21271720526931803, -0.6921278738183232, -0.027798974395459197, 0.6156221865413001, -0.943844843229722, -0.37814338753656984, -0.346697561288283, 0.959689866112815, -0.6581327285953573, 0.45320215778348927, 0.6343899166677849, 0.7184500189472003, 0.31473439281736426, 0.909850953825984, 0.327494113344021, -0.6859820460379871, 0.698390613604543, 0.36895061782154226, -0.40700807314162013, -0.8564584475054688, 0.9512159237701696, 0.38003861810208317, -0.6591048074496089, 0.31602168515075246, 0.5163556885933733, 0.22455111897319657, 0.6860373701494031, -0.19143931728593455, 0.5347486025502406, 0.30685861674958614, 0.6680106295594315, -0.7170044666316511, -0.5610270137903806, 0.9712418886036154, 0.7634094256975745, -0.5480966738938566, 0.3785145377791381, -0.630803254174364, 0.6342967766025949, 0.44125224735357094, 0.8478716555095993, 0.5534841711179159, 0.7705977685299286, -0.18930446642110654, 0.23094556935002641, -0.24983785152770066, -0.11198839085253298, -0.42475286427381853, 0.707850020350413, -0.7532211388237622, -0.05156112113782596, -0.9422975195355956, -0.04993010363516004, 0.5543207715527336, -0.21916427612017908, -0.07232634005274563, 0.20158843070979016, -0.0284213025674267, 0.7117521434762326, -0.15504515740341995, -0.3543303808507028, 0.14171308900720647, 0.32657141888383445, -0.2589601193621993, 0.683014686796598, -0.22746652823845537, -0.7309305979122742, -0.02281071727110584, -0.4644920350522532]

    matrixA = np.array(matrixA)
    matrixC = np.array(matrixC)
    matrixA = matrixA.reshape((1, matrixA.size))
    matrixC = matrixC.reshape((matrixC.size, 1))

    '''Contruct BatchPlan'''
    myBatchPlan.fromMatrix(matrixA, True)
    # myBatchPlan.matrixAdd([matrixB], [False])
    myBatchPlan.matrixMul([matrixC])
    print("\n-------------------Batch Plan before weave:-------------------")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Batch Plan after weave:-------------------")
    myBatchPlan.weave()
    batch_scheme = myBatchPlan.getBatchScheme()
    max_element_num, split_num = batch_scheme[0]
    print("Element num in one vector: ", + max_element_num)
    print("Split num: ", + split_num)
    print("Memory of each slot: ", + myBatchPlan.encoder.slot_mem_size)
    print("Memory of extra sign bits: ", + myBatchPlan.encoder.sign_bits)

    '''Encrypt'''
    print("\n-------------------Encryption:-------------------")
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()
    encrypted_row_vec = myBatchPlan.encrypt(matrixA, batch_scheme[0], encrypter.public_key)

    '''Assign encrypted vector'''
    myBatchPlan.assignEncryptedVector(0, 0, encrypted_row_vec)

    print("\n-------------------Begin to exec Batch Plan.-------------------")
    outputs = myBatchPlan.parallelExec()
    '''Decrypt & shift sum'''
    res = []
    for output in outputs:
        # each output represent the output of one root node
        row_vec = []
        for element in output:
            real_res = 0
            for batch_encrypted_number_idx in range(len(element)):
                temp = myBatchPlan.decrypt(element[batch_encrypted_number_idx], encrypter.privacy_key)
                real_res += temp[batch_encrypted_number_idx]
            row_vec.append(real_res)
        res.append(row_vec)
    outputs = res

    row_num, col_num = myBatchPlan.matrix_shape
    output_matrix = np.zeros(myBatchPlan.matrix_shape)
    for row_id in range(row_num):
        output_matrix[row_id, :] = outputs[row_id][0:col_num]
    print("\n-------------------Batch Plan output:-------------------")
    print(output_matrix)
    print("\n-------------------Numpy output:-------------------")
    result = (matrixA).dot(matrixC)
    print(result)
    if np.allclose(output_matrix, result):
        print("\n-------------------Test Pass!-------------------")
    else:
        print("\n-------------------Test Fail-------------------")
        print(output_matrix == result)
        # print(matrixA.reshape(matrixA.size).tolist())
        # print(matrixC.reshape(matrixA.size).tolist())

def scalar_mul():
    encoder = BatchEncoder(1, 64, 256, 64, 3)        # encode [-1, 1] using 64 bits
    row_vec_A = np.random.uniform(-1, 0, 3)
    row_vec_B = np.random.uniform(-1, 0, 3)
    row_vec_A = np.array([-1, -1, -1])
    print("----------------Original vector:----------------")
    print(row_vec_A)
    # print(row_vec_B)
    print("----------------Encode:----------------")
    batch_encode_A = encoder.batchEncode(row_vec_A)
    # scalar_encode_B = encoder.scalarEncode(row_vec_B)
    print("encode A: ", '0x%x'%batch_encode_A)
    # print("scalar B: ")
    # for scalar in scalar_encode_B:
    #     print('0x%x'%scalar)
    print("----------------Encrypt:----------------")
    key_generator = PaillierEncrypt()
    key_generator.generate_key()
    encrypter = BatchEncryption()
    encrypted_A = encrypter.gpuBatchEncrypt([batch_encode_A], encoder.scaling, encoder.size, key_generator.public_key)
    # shift
    encrypted_A.value = encrypted_A.value.mul_with_big_integer(int(pow(2, encoder.slot_mem_size)))
    encrypted_A.value = encrypted_A.value.mul_with_big_integer(int(pow(2, encoder.slot_mem_size)))

    print("----------------Decrypt:----------------")
    decrypted_A = encrypter.gpuBatchDecrypt(encrypted_A, key_generator.privacy_key)
    decrypted_A[0] = decrypted_A[0] >> encoder.slot_mem_size
    decrypted_A[0] = decrypted_A[0] >> encoder.slot_mem_size
    print("encode A: ", '0x%x'%decrypted_A[0])

def lr_procedure():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=32)
    self_fore_gradient = np.random.uniform(-1, 1, (1, 300))     # ciphertext
    host_fore_gradient = np.random.uniform(-1, 1, (1, 300))
    self_feature = np.random.uniform(-1, 1, (300, 20))     # plaintext

    '''Contruct BatchPlan'''
    myBatchPlan.fromMatrix(self_fore_gradient, True)
    myBatchPlan.matrixAdd([host_fore_gradient], [False])
    fore_gradient_node = myBatchPlan.root_nodes[0]
    myBatchPlan.matrixMul([self_feature])
    print("\n-------------------Batch Plan before weave:-------------------")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Batch Plan after weave:-------------------")
    myBatchPlan.weave()
    batch_scheme = myBatchPlan.getBatchScheme()
    max_element_num, split_num = batch_scheme[0]
    print("Element num in one vector: ", + max_element_num)
    print("Split num: ", + split_num)
    print("Memory of each slot: ", + myBatchPlan.encoder.slot_mem_size)
    print("Memory of extra sign bits: ", + myBatchPlan.encoder.sign_bits)

    '''Encrypt'''
    print("\n-------------------Encryption:-------------------")
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()
    encrypted_row_vec = myBatchPlan.encrypt(self_fore_gradient, batch_scheme[0], encrypter.public_key)

    '''Assign encrypted vector'''
    myBatchPlan.assignEncryptedVector(0, 0, encrypted_row_vec)

    print("\n-------------------Begin to exec Batch Plan.-------------------")
    outputs = myBatchPlan.parallelExec()
    '''Decrypt & shift sum'''
    res = []
    for output in outputs:
        # each output represent the output of one root node
        row_vec = []
        for element in output:
            real_res = 0
            for batch_encrypted_number_idx in range(len(element)):
                temp = myBatchPlan.decrypt(element[batch_encrypted_number_idx], encrypter.privacy_key)
                real_res += temp[batch_encrypted_number_idx]
            row_vec.append(real_res)
        res.append(row_vec)
    outputs = res
    '''Calculate bias'''
    bias_middle_grad = fore_gradient_node.getBatchData()
    bias_middle_grad.value = bias_middle_grad.value.sum()
    bias_grad = sum(myBatchPlan.decrypt(bias_middle_grad, encrypter.privacy_key))

    row_num, col_num = myBatchPlan.matrix_shape
    output_matrix = np.zeros(myBatchPlan.matrix_shape)
    for row_id in range(row_num):
        output_matrix[row_id, :] = outputs[row_id][0:col_num]
    print("\n-------------------Batch Plan output:-------------------")
    print("unilateral_gradient: ", output_matrix)
    print("bias gradient: ", bias_grad)
    print("\n-------------------Numpy output:-------------------")
    result = (self_fore_gradient+host_fore_gradient).dot(self_feature)
    plain_bias = (self_fore_gradient+host_fore_gradient).sum()
    print(result)
    print(plain_bias)
    if np.allclose(output_matrix, result):
        print("\n-------------------Test Pass!-------------------")
    else:
        print("\n-------------------Test Fail-------------------")
        print(output_matrix == result)

encrypted_mul()
