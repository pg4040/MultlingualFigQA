new_val = "1080,1\n1081,1\n1082,0\n1083,1\n1084,0\n1085,1\n1086,1\n1087,1\n1088,1\n1089,1\n1090,0\n1091,1\n1092,0\n1093,0\n1094,1\n1095,1\n1096,0\n1097,1\n1098,0\n1099,1\n1100,0\n1101,1\n1102,0\n1103,1\n1104,0\n1105,1\n1106,0\n1107,1\n1108,1\n1109,1\n1110,1\n1111,1\n1112,0\n1113,1\n1114,0\n1115,1\n1116,1\n1117,1\n1118,1\n1119,1\n1120,1\n1121,1\n1122,0\n1123,1\n1124,1\n1125,1\n1126,0\n1127,1\n1128,0\n1129,1\n1130,1\n1131,0\n1132,0\n1133,1\n1134,0\n1135,1\n1136,1\n1137,1\n1138,0\n1139,1"
import pandas as pd
in_csv = './outputs/gpt_response_id_batch.csv'
in_df = pd.read_csv(in_csv)
last_row_index = in_df.index[-1]
last_col_index = in_df.columns[-1]
in_df.at[last_row_index, last_col_index] = new_val
in_df.to_csv(in_csv)
