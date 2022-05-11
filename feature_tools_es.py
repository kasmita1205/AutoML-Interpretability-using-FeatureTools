import numpy as np
import pandas as pd
import featuretools as ft

courses = pd.read_csv("OULAD/courses.csv")
assessments = pd.read_csv("OULAD/assessments.csv")
vle = pd.read_csv("OULAD/vle.csv")
studentInfo = pd.read_csv("OULAD/studentInfo.csv")
studentRegistration = pd.read_csv("OULAD/studentRegistration.csv")
studentAssessment = pd.read_csv("OULAD/studentAssessment.csv")
studentVle = pd.read_csv("OULAD/studentVle.csv")
courses['courseKey'] = courses['code_module']+courses['code_presentation']
assessments['courseKey'] = assessments['code_module']+assessments['code_presentation']
vle['courseKey'] = vle['code_module']+ vle['code_presentation']
studentInfo['courseKey'] = studentInfo['code_module']+studentInfo['code_presentation']
studentRegistration['courseKey'] = studentRegistration['code_module']+studentRegistration['code_presentation']
studentVle['courseKey'] = studentVle['code_module']+studentVle['code_presentation']
studentDetails = studentInfo[["id_student","gender","disability"]]
studentDetails = studentDetails.drop_duplicates()


studentVle_sample = studentVle.sample(frac=0.2, random_state=1)

es = ft.EntitySet(id="evaluation")
es = es.entity_from_dataframe(entity_id="courses", dataframe=courses,index="courseKey",)
es = es.entity_from_dataframe(entity_id="assessments", dataframe=assessments, index="id_assessment",)
es = es.entity_from_dataframe(entity_id="vle",dataframe=vle,index="id_site")
es = es.entity_from_dataframe(entity_id="studentInfo", dataframe= studentInfo, index = "studentInfo_index",make_index=True)
es = es.entity_from_dataframe(entity_id="studentRegistration", dataframe=studentRegistration, index = "studentRegistration_index",make_index=True)
es = es.entity_from_dataframe(entity_id="studentAssessment", dataframe= studentAssessment, index = "studentAssessment_index",make_index=True)
es = es.entity_from_dataframe(entity_id="studentVle", dataframe= studentVle_sample, index = "studentVle_index",make_index=True)
es = es.entity_from_dataframe(entity_id="studentDetails", dataframe=studentDetails, index = "id_student")

course_to_assessment = ft.Relationship(es["courses"]["courseKey"],es["assessments"]["courseKey"])
course_to_vle = ft.Relationship(es["courses"]["courseKey"],es["vle"]["courseKey"])
course_to_studentInfo = ft.Relationship(es["courses"]["courseKey"],es["studentInfo"]["courseKey"])
course_to_studentRegistration = ft.Relationship(es["courses"]["courseKey"],es["studentRegistration"]["courseKey"])
course_to_studentVle= ft.Relationship(es["courses"]["courseKey"],es["studentVle"]["courseKey"])
assessment_to_studentAssessment = ft.Relationship(es["assessments"]["id_assessment"],es["studentAssessment"]["id_assessment"])
vle_to_studentVle = ft.Relationship(es["vle"]["id_site"],es["studentVle"]["id_site"])
studentDetails_to_studentInfo = ft.Relationship(es["studentDetails"]["id_student"], es["studentInfo"]["id_student"])
studentDetails_to_studentRegistration = ft.Relationship(es["studentDetails"]["id_student"], es["studentRegistration"]["id_student"])
studentDetails_to_studentAssessment = ft.Relationship(es["studentDetails"]["id_student"], es["studentAssessment"]["id_student"])
studentDetails_to_studentVle = ft.Relationship(es["studentDetails"]["id_student"], es["studentVle"]["id_student"])

es = es.add_relationship(course_to_assessment)
es =es.add_relationship(course_to_vle)
es =es.add_relationship(course_to_studentInfo)
es=es.add_relationship(course_to_studentVle)
es =es.add_relationship(course_to_studentRegistration)
es= es.add_relationship(assessment_to_studentAssessment)
es= es.add_relationship(vle_to_studentVle)
es = es.add_relationship(studentDetails_to_studentInfo)
es = es.add_relationship(studentDetails_to_studentAssessment)
es = es.add_relationship(studentDetails_to_studentRegistration)
es = es.add_relationship(studentDetails_to_studentVle)

0
aggregation_primitives = [
    'max',
    'median',
    'mode',
    'time_since_first',
    'sum',
    'avg_time_between',
    'num_unique',
    'skew',
    'min',
    'trend',
    'mean',
    'count',
    'time_since_last',
    'std',
    'entropy',
]
transform_primitives = [
    'time_since_previous',
    'divide_by_feature',
    'greater_than_equal_to',
    'time_since',
    'cum_min',
    'cum_count',
    'month',
    'cum_max',
    'cum_mean',
    'weekday',
    'cum_sum',
    'percentile',
]

#es.plot()
feature_matrix, feature_defs = ft.dfs(entityset=es,target_entity="studentDetails", agg_primitives=aggregation_primitives,
                                      trans_primitives=transform_primitives)
print(feature_matrix.shape)
print(feature_matrix.iloc[:10,[140,141]])
feature_def = feature_defs[141]
print(ft.describe_feature(feature_def))

#ft.graph_feature(feature_def)
feature_matrix.to_csv("features.csv")
