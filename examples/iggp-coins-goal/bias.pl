max_body(7).

head_pred(goal,3).
body_pred(my_true_cell,3).
body_pred(role,1).
body_pred(my_pos,1).
body_pred(c_zerocoins,1).
body_pred(c_onecoin,1).
body_pred(c_twocoins,1).
body_pred(my_true_step,2).

body_pred(pos_1,1).
body_pred(pos_2,1).
body_pred(pos_3,1).
body_pred(pos_4,1).
body_pred(pos_5,1).
body_pred(pos_6,1).
body_pred(pos_7,1).
body_pred(pos_8,1).

body_pred(score_0,1).
body_pred(score_1,1).
body_pred(score_2,1).
body_pred(score_3,1).
body_pred(score_4,1).
body_pred(score_5,1).
body_pred(score_6,1).
body_pred(score_7,1).
body_pred(score_8,1).
body_pred(score_9,1).
body_pred(score_10,1).
body_pred(score_11,1).
body_pred(score_12,1).
body_pred(score_13,1).
body_pred(score_14,1).
body_pred(score_15,1).
body_pred(score_16,1).
body_pred(score_17,1).
body_pred(score_18,1).
body_pred(score_19,1).
body_pred(score_20,1).
body_pred(score_21,1).
body_pred(score_22,1).
body_pred(score_23,1).
body_pred(score_24,1).
body_pred(score_25,1).
body_pred(score_26,1).
body_pred(score_27,1).
body_pred(score_28,1).
body_pred(score_29,1).
body_pred(score_30,1).
body_pred(score_31,1).
body_pred(score_32,1).
body_pred(score_33,1).
body_pred(score_34,1).
body_pred(score_35,1).
body_pred(score_36,1).
body_pred(score_37,1).
body_pred(score_38,1).
body_pred(score_39,1).
body_pred(score_40,1).
body_pred(score_41,1).
body_pred(score_42,1).
body_pred(score_43,1).
body_pred(score_44,1).
body_pred(score_45,1).
body_pred(score_46,1).
body_pred(score_47,1).
body_pred(score_48,1).
body_pred(score_49,1).
body_pred(score_50,1).
body_pred(score_51,1).
body_pred(score_52,1).
body_pred(score_53,1).
body_pred(score_54,1).
body_pred(score_55,1).
body_pred(score_56,1).
body_pred(score_57,1).
body_pred(score_58,1).
body_pred(score_59,1).
body_pred(score_60,1).
body_pred(score_61,1).
body_pred(score_62,1).
body_pred(score_63,1).
body_pred(score_64,1).
body_pred(score_65,1).
body_pred(score_66,1).
body_pred(score_67,1).
body_pred(score_68,1).
body_pred(score_69,1).
body_pred(score_70,1).
body_pred(score_71,1).
body_pred(score_72,1).
body_pred(score_73,1).
body_pred(score_74,1).
body_pred(score_75,1).
body_pred(score_76,1).
body_pred(score_77,1).
body_pred(score_78,1).
body_pred(score_79,1).
body_pred(score_80,1).
body_pred(score_81,1).
body_pred(score_82,1).
body_pred(score_83,1).
body_pred(score_84,1).
body_pred(score_85,1).
body_pred(score_86,1).
body_pred(score_87,1).
body_pred(score_88,1).
body_pred(score_89,1).
body_pred(score_90,1).
body_pred(score_91,1).
body_pred(score_92,1).
body_pred(score_93,1).
body_pred(score_94,1).
body_pred(score_95,1).
body_pred(score_96,1).
body_pred(score_97,1).
body_pred(score_98,1).
body_pred(score_99,1).
body_pred(score_100,1).

%% BECAUSE WE DO NOT LEARN FROM INTERPRETATIONS
:-
    clause(C),
    #count{V : clause_var(C,V),var_type(C,V,ex)} != 1.

type(goal,(ex,agent,score)).
type(does_jump,(ex,agent,pos,pos)).
type(my_succ,(pos,pos)).
type(my_true_cell,(ex,pos,cell_value)).
type(role,(agent,)).
type(my_pos,(pos,)).
type(c_zerocoins,(cell_value,)).
type(c_onecoin,(cell_value,)).
type(c_twocoins,(cell_value,)).
type(my_true_step,(ex, pos)).

type(pos_1,(pos,)).
type(pos_2,(pos,)).
type(pos_3,(pos,)).
type(pos_4,(pos,)).
type(pos_5,(pos,)).
type(pos_6,(pos,)).
type(pos_7,(pos,)).
type(pos_8,(pos,)).

type(score_0,(score,)).
type(score_1,(score,)).
type(score_2,(score,)).
type(score_3,(score,)).
type(score_4,(score,)).
type(score_5,(score,)).
type(score_6,(score,)).
type(score_7,(score,)).
type(score_8,(score,)).
type(score_9,(score,)).
type(score_10,(score,)).
type(score_11,(score,)).
type(score_12,(score,)).
type(score_13,(score,)).
type(score_14,(score,)).
type(score_15,(score,)).
type(score_16,(score,)).
type(score_17,(score,)).
type(score_18,(score,)).
type(score_19,(score,)).
type(score_20,(score,)).
type(score_21,(score,)).
type(score_22,(score,)).
type(score_23,(score,)).
type(score_24,(score,)).
type(score_25,(score,)).
type(score_26,(score,)).
type(score_27,(score,)).
type(score_28,(score,)).
type(score_29,(score,)).
type(score_30,(score,)).
type(score_31,(score,)).
type(score_32,(score,)).
type(score_33,(score,)).
type(score_34,(score,)).
type(score_35,(score,)).
type(score_36,(score,)).
type(score_37,(score,)).
type(score_38,(score,)).
type(score_39,(score,)).
type(score_40,(score,)).
type(score_41,(score,)).
type(score_42,(score,)).
type(score_43,(score,)).
type(score_44,(score,)).
type(score_45,(score,)).
type(score_46,(score,)).
type(score_47,(score,)).
type(score_48,(score,)).
type(score_49,(score,)).
type(score_50,(score,)).
type(score_51,(score,)).
type(score_52,(score,)).
type(score_53,(score,)).
type(score_54,(score,)).
type(score_55,(score,)).
type(score_56,(score,)).
type(score_57,(score,)).
type(score_58,(score,)).
type(score_59,(score,)).
type(score_60,(score,)).
type(score_61,(score,)).
type(score_62,(score,)).
type(score_63,(score,)).
type(score_64,(score,)).
type(score_65,(score,)).
type(score_66,(score,)).
type(score_67,(score,)).
type(score_68,(score,)).
type(score_69,(score,)).
type(score_70,(score,)).
type(score_71,(score,)).
type(score_72,(score,)).
type(score_73,(score,)).
type(score_74,(score,)).
type(score_75,(score,)).
type(score_76,(score,)).
type(score_77,(score,)).
type(score_78,(score,)).
type(score_79,(score,)).
type(score_80,(score,)).
type(score_81,(score,)).
type(score_82,(score,)).
type(score_83,(score,)).
type(score_84,(score,)).
type(score_85,(score,)).
type(score_86,(score,)).
type(score_87,(score,)).
type(score_88,(score,)).
type(score_89,(score,)).
type(score_90,(score,)).
type(score_91,(score,)).
type(score_92,(score,)).
type(score_93,(score,)).
type(score_94,(score,)).
type(score_95,(score,)).
type(score_96,(score,)).
type(score_97,(score,)).
type(score_98,(score,)).
type(score_99,(score,)).
type(score_100,(score,)).

