import backoff
import openai
from openai import OpenAI
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import heapq
import random

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.Timeout,
        openai.APIError,
        openai.APIConnectionError,
        openai.APIStatusError,
    ),
)
def get_precise_response(messages, model="gpt-4o-2024-08-06", temperature=0.2, top_p=0.1):
    message = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
    )
    return message.choices[0].message.content


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.Timeout,
        openai.APIError,
        openai.APIConnectionError,
        openai.APIStatusError,
    ),
)
def get_chatbot_response(
    messages, model="gpt-4o-2024-08-06", temperature=0.7, top_p=0.8, max_tokens=100
):
    message = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=100,
    )
    return message.choices[0].message.content


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.Timeout,
        openai.APIError,
        openai.APIConnectionError,
        openai.APIStatusError,
    ),
)
def get_json_response(messages, model="gpt-4o-2024-08-06", temperature=0.2, top_p=0.1):
    message = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        response_format={"type": "json_object"},
    )
    return message.choices[0].message.content


stage2description = {
    "Precontemplation": "The client doesn't think their behavior is problematic.",
    "Contemplation": "The client feels that their behavior is problematic, but still hesitate whether to change.",
    "Preparation": "The client begins discuss about steps toward behavior change.",
}


class Client:
    def __init__(
        self,
        goal,
        behavior,
        reference,
        personas,
        initial_stage,
        final_stage,
        motivation,
        beliefs,
        plans,
        receptivity,
        model,
        wikipedia_dir,
        retriever_path
    ):
        self.goal = goal
        self.behavior = behavior
        self.personas = personas
        self.motivation = motivation[-1]
        self.engagemented_topics = motivation[:-1]
        self.beliefs = beliefs
        self.initial_stage = initial_stage
        self.state = initial_stage
        self.final_stage = final_stage
        self.acceptable_plans = plans
        self.receptivity = receptivity
        self.engagement = receptivity
        self.context = [
            "Counselor: Hello. How are you?",
            "Client: I am good. What about you?",
        ]
        self.action2prompt = {
            "Deny": "You should directly refuse to admit your behavior is problematic or needs change.",
            "Downplay": "You should downplay the importance or impact of your behavior.",
            "Blame": "You should blame external factors or others to justify your behavior.",
            "Inform": "You should share details about your background, experiences, or emotions revealing the current state.",
            "Engage": "You should interact with counselor consistently based on your state and mimic the style in the reference conversation.",
            "Hesitate": "You should show uncertainty, indicating ambivalence about change.",
            "Doubt": "You should expresse skepticism about the practicality or success of proposed changes but not reveal further information.",
            "Acknowledge": "You should acknowledge the need for change.",
            "Accept": "You should agree to adopt the suggested action plan.",
            "Reject": "You should decline the proposed plan, deeming it unsuitable.",
            "Plan": "You should propose or detail steps for a change plan.",
            "Terminate": "You should highlight current state and engagement, express a desire to end the current session, and suggest further discussion be deferred to a later time.",
        }

        self.state2prompt = {
            "Precontemplation": f"You doesn't think your {self.behavior} is problematic and wants to sustain.",
            "Contemplation": f"You feels that your {self.behavior} is problematic, but still hesitate about {self.goal}.",
            "Preparation": "You gets ready to take action to change and begins discuss about steps toward {self.goal}.",
        }

        self.topic2description = {
            "Infection": f"When discussing [topic], The counselor may discuss how {self.behavior} can weaken your immune system, making you more susceptible to infections. They may also talk about how {self.goal} can help strengthen your immune system and reduce your risk of frequent or severe infections.",
            "Hypertension": f"When discussing [topic], The counselor may explore how {self.behavior} can contribute to increased blood pressure and raise your risk of hypertension. They may also discuss how {self.goal} can help maintain a healthier blood pressure level and lower your risk of complications from hypertension.",
            "Flu": f"When discussing [topic], The counselor may explain how {self.behavior} can increase your chances of contracting the flu or experiencing more severe symptoms. They may also highlight how {self.goal} can improve your immune defense, reducing your likelihood of catching the flu or lessening its impact.",
            "Inflammation": f"When discussing [topic], The counselor may address how {self.behavior} can lead to chronic inflammation in your body, increasing your risk for related health conditions. They may also discuss how {self.goal} can help reduce inflammation and promote better long-term health.",
            "Liver Disease": f"When discussing [topic], The counselor may explore how {self.behavior} can cause liver damage, increasing your risk of liver disease. They may also discuss how {self.goal} can protect your liver, prevent damage, and lower your chances of developing serious liver conditions.",
            "Lung Cancer": f"When discussing [topic], The counselor may discuss how {self.behavior} can increase your risk of developing lung cancer. They may also highlight how {self.goal} can reduce your risk of lung cancer and improve your overall respiratory health.",
            "COPD": f"When discussing [topic], The counselor may explore how {self.behavior} can worsen symptoms of COPD, making it harder for you to breathe. They may also explain how {self.goal} can improve your lung function and help manage COPD symptoms.",
            "Asthma": f"When discussing [topic], The counselor may explain how {self.behavior} can trigger or worsen asthma attacks. They may also discuss how {self.goal} can help control asthma symptoms and improve your ability to manage the condition.",
            "Stroke": f"When discussing [topic], The counselor may address how {self.behavior} can increase your risk of stroke by negatively impacting your cardiovascular health. They may also highlight how {self.goal} can help reduce this risk and support a healthier cardiovascular system.",
            "Diabetes": f"When discussing [topic], The counselor may explore how {self.behavior} can contribute to the development or worsening of diabetes. They may also discuss how {self.goal} can help manage your blood sugar levels and reduce the risk of diabetes-related complications.",
            "Physical Activity": f"When discussing [topic], The counselor may discuss how {self.behavior} can reduce your physical activity levels, increasing your risk of various health problems. They may also explain how {self.goal} can boost your physical activity, improving your fitness and overall health.",
            "Sport": f"When discussing [topic], The counselor may explain how {self.behavior} can decrease your performance or participation in sports. They may also discuss how {self.goal} can enhance your physical conditioning and boost your ability to engage in sporting activities.",
            "Physical Fitness": f"When discussing [topic], The counselor may discuss how {self.behavior} can negatively affect your physical fitness, reducing your overall health. They may also explain how {self.goal} can improve your fitness level and contribute to better health and well-being.",
            "Strength": f"When discussing [topic], The counselor may explore how {self.behavior} can lead to a loss of strength, making everyday tasks more difficult. They may also discuss how {self.goal} can help you build or regain muscle strength.",
            "Flexibility": f"When discussing [topic], The counselor may explain how {self.behavior} can reduce your flexibility, causing stiffness or discomfort. They may also highlight how {self.goal} can improve your flexibility, making movement easier and reducing the risk of injury.",
            "Endurance": f"When discussing [topic], The counselor may discuss how {self.behavior} can lower your endurance, making it harder to sustain physical activities for long periods. They may also explore how {self.goal} can improve your stamina and overall physical endurance.",
            "Dentistry": f"When discussing [topic], The counselor may address how {self.behavior} can lead to poor dental hygiene, increasing your risk of cavities or gum disease. They may also explain how {self.goal} can improve your oral health and reduce the likelihood of dental problems.",
            "Caregiver Burden": f"When discussing [topic], The counselor may explore how {self.behavior} can place additional stress on caregivers, leading to burnout or reduced quality of care. They may also discuss how {self.goal} can alleviate the burden on caregivers and improve the care provided.",
            "Independent Living": f"When discussing [topic], The counselor may explain how {self.behavior} can limit your ability to live independently, making you more reliant on others. They may also discuss how {self.goal} can help you regain independence and improve your quality of life.",
            "Human Appearance": f"When discussing [topic], The counselor may address how {self.behavior} can negatively impact your appearance, such as causing skin issues or weight gain. They may also explain how {self.goal} can improve your physical appearance and boost your self-confidence.",
            "Depression": f"When discussing [topic], The counselor may explore how {self.behavior} can worsen symptoms of depression, affecting your mood and daily life. They may also discuss how {self.goal} can improve your emotional well-being and reduce the impact of depression.",
            "Chronodisruption": f"When discussing [topic], The counselor may explain how {self.behavior} can disrupt your body’s natural rhythms, leading to sleep problems or fatigue. They may also highlight how {self.goal} can restore healthy sleep patterns and improve your overall well-being.",
            "Anxiety Disorders": f"When discussing [topic], The counselor may explore how {self.behavior} can increase your anxiety, leading to stress and panic attacks. They may also discuss how {self.goal} can help manage anxiety and promote emotional stability.",
            "Cognitive Decline": f"When discussing [topic], The counselor may discuss how {self.behavior} can accelerate cognitive decline, affecting your memory and thinking abilities. They may also highlight how {self.goal} can help protect brain function and slow cognitive deterioration.",
            "Safe Sex": f"When discussing [topic], The counselor may explain how {self.behavior} can increase your risk of sexually transmitted infections or unintended pregnancies. They may also discuss how {self.goal} can promote safer sexual practices and reduce these risks.",
            "Maternal Health": f"When discussing [topic], The counselor may address how {self.behavior} can negatively affect your health during pregnancy, increasing the risk of complications. They may also discuss how {self.goal} can support a healthier pregnancy and reduce the likelihood of problems during childbirth.",
            "Preterm Birth": f"When discussing [topic], The counselor may explore how {self.behavior} can increase the risk of preterm birth, leading to complications for both mother and baby. They may also highlight how {self.goal} can help ensure a full-term, healthy pregnancy.",
            "Miscarriage": f"When discussing [topic], The counselor may explain how {self.behavior} can increase the risk of miscarriage, causing emotional and physical distress. They may also discuss how {self.goal} can help support a healthy pregnancy and reduce miscarriage risk.",
            "Birth Defects": f"When discussing [topic], The counselor may address how {self.behavior} can raise the risk of birth defects during pregnancy. They may also explain how {self.goal} can promote a healthier pregnancy and reduce the risk of complications.",
            "Productivity": f"When discussing [topic], The counselor may explore how {self.behavior} can negatively affect your productivity at work, making it harder to perform well. They may also discuss how {self.goal} can improve your focus and efficiency.",
            "Absenteeism": f"When discussing [topic], The counselor may explain how {self.behavior} can increase absenteeism, causing you to miss work or other responsibilities. They may also highlight how {self.goal} can help you be more consistent and present in your daily life.",
            "Workplace Relationships": f"When discussing [topic], The counselor may discuss how {self.behavior} can strain relationships with colleagues, leading to conflicts at work. They may also explain how {self.goal} can help improve communication and foster positive workplace relationships.",
            "Career Break": f"When discussing [topic], The counselor may address how {self.behavior} can lead to career interruptions or breaks, affecting your professional progress. They may also discuss how {self.goal} can help you maintain continuity in your career.",
            "Career Assessment": f"When discussing [topic], The counselor may explain how {self.behavior} can affect career assessments, leading to negative evaluations or feedback. They may also highlight how {self.goal} can improve your performance and result in more favorable assessments.",
            "Absence Rate": f"When discussing [topic], The counselor may explore how {self.behavior} can increase your absence rate at work, impacting your job security. They may also discuss how {self.goal} can help you reduce absences and improve your work attendance.",
            "Salary": f"When discussing [topic], The counselor may address how {self.behavior} can hinder salary progression, limiting your earning potential. They may also discuss how {self.goal} can help you increase your salary and financial stability.",
            "Workplace Wellness": f"When discussing [topic], The counselor may discuss how {self.behavior} can undermine workplace wellness initiatives, affecting your health at work. They may also explain how {self.goal} can help you benefit from wellness programs and improve your job satisfaction.",
            "Workplace Incivility": f"When discussing [topic], The counselor may explore how {self.behavior} can contribute to incivility in the workplace, leading to a negative environment. They may also highlight how {self.goal} can help promote respect and cooperation in your work setting.",
            "Cost of Living": f"When discussing [topic], The counselor may discuss how {self.behavior} can make it harder to manage the cost of living, leading to financial strain. They may also explain how {self.goal} can help improve your financial situation and reduce stress related to expenses.",
            "Personal Budget": f"When discussing [topic], The counselor may address how {self.behavior} can make it difficult to stick to a personal budget, leading to financial instability. They may also discuss how {self.goal} can help you manage your finances more effectively.",
            "Debt": f"When discussing [topic], The counselor may explain how {self.behavior} can lead to increased debt, affecting your financial security. They may also discuss how {self.goal} can help you reduce debt and improve your financial well-being.",
            "Income Deficit": f"When discussing [topic], The counselor may explore how {self.behavior} can contribute to income deficits, making it harder to cover basic expenses. They may also explain how {self.goal} can help improve your financial management and income stability.",
            "Family Estrangement": f"When discussing [topic], The counselor may discuss how {self.behavior} can lead to emotional or physical distance between family members. They may also highlight how {self.goal} can help repair family relationships and promote reconciliation.",
            "Family Disruption": f"When discussing [topic], The counselor may explain how {self.behavior} can disrupt family dynamics, leading to conflict or instability. They may also discuss how {self.goal} can strengthen family bonds and promote harmony.",
            "Divorce": f"When discussing [topic], The counselor may address how {self.behavior} can contribute to marital conflict, potentially leading to divorce. They may also discuss how {self.goal} can improve communication and reduce the risk of divorce.",
            "Role Model": f"When discussing [topic], The counselor may explain how {self.behavior} can affect your ability to serve as a positive role model, particularly for your children. They may also highlight how {self.goal} can help you set a better example for those who look up to you.",
            "Child Development": f"When discussing [topic], The counselor may discuss how {self.behavior} can impact your child’s emotional, social, or cognitive development. They may also explain how {self.goal} can support healthier growth and development in your child.",
            "Paternal Bond": f"When discussing [topic], The counselor may explore how {self.behavior} can weaken the bond between you and your child, affecting your relationship. They may also discuss how {self.goal} can help strengthen this bond and promote a closer connection.",
            "Child Care": f"When discussing [topic], The counselor may explain how {self.behavior} can interfere with your ability to provide consistent child care. They may also highlight how {self.goal} can help you offer more stable and nurturing care for your child.",
            "Habituation": f"When discussing [topic], The counselor may discuss how {self.behavior} can negatively affect a child’s ability to develop healthy habits or adapt to new environments. They may also explain how {self.goal} can support positive habituation and learning.",
            "Arrest": f"When discussing [topic], The counselor may address how {self.behavior} can increase your risk of arrest, leading to legal trouble. They may also highlight how {self.goal} can help you avoid legal issues and stay on the right side of the law.",
            "Imprisonment": f"When discussing [topic], The counselor may explain how {self.behavior} can lead to imprisonment, causing long-term legal and social consequences. They may also discuss how {self.goal} can help you avoid incarceration and promote responsible behavior.",
            "Child Custody": f"When discussing [topic], The counselor may explore how {self.behavior} can affect your ability to maintain child custody, leading to legal challenges. They may also discuss how {self.goal} can help you improve your parenting abilities and secure your custody rights.",
            "Traffic Ticket": f"When discussing [topic], The counselor may address how {self.behavior} can increase your chances of receiving traffic tickets or other legal penalties. They may also discuss how {self.goal} can help you adopt safer driving practices and avoid infractions.",
            "Complaint": f"When discussing [topic], The counselor may explain how {self.behavior} can result in legal complaints or disputes. They may also highlight how {self.goal} can help you reduce conflicts and maintain positive relationships with others.",
            "Attendance": f"When discussing [topic], The counselor may explore how {self.behavior} can affect your attendance, causing you to miss school or other responsibilities. They may also discuss how {self.goal} can help you improve your attendance and stay on track.",
            "Suspension": f"When discussing [topic], The counselor may explain how {self.behavior} can lead to suspension from school, affecting your academic progress. They may also highlight how {self.goal} can help you avoid disciplinary actions and stay engaged in school.",
            "Scholarship": f"When discussing [topic], The counselor may address how {self.behavior} can reduce your eligibility for scholarships, limiting academic opportunities. They may also discuss how {self.goal} can help improve your academic performance and increase your chances of receiving scholarships.",
            "Exam": f"When discussing [topic], The counselor may discuss how {self.behavior} can negatively impact your exam preparation and performance, leading to lower grades. They may also explain how {self.goal} can help improve your focus and exam results.",
            "Health": f"When discussing [topic], The counselor may talk about how {self.behavior} affects your overall health, including physical, mental, and emotional well-being. They may focus on specific health conditions, fitness levels, mental health, or healthcare practices. The counselor can also discuss how {self.goal} can improve your health and reduce risks.",
            "Diseases": f"When discussing [topic], The counselor may talk about how {self.behavior} leads to specific health conditions, such as infections, hypertension, flu, inflammation, liver disease, lung cancer, chronic obstructive pulmonary disease (COPD), asthma, stroke, and diabetes. The counselor can also discuss how {self.goal} can help prevent or manage these diseases.",
            "Fitness": f"When discussing [topic], The counselor may talk about how {self.behavior} reduces your physical activity, strength, flexibility, or endurance. The counselor can also discuss how {self.goal} can improve your fitness and overall physical health.",
            "Health Care": f"When discussing [topic], The counselor may talk about how {self.behavior} affects your ability to maintain personal healthcare, such as dental hygiene, caregiver burden, independent living, or human appearance. The counselor can also discuss how {self.goal} can improve your self-care and healthcare routines.",
            "Mental Disorders": f"When discussing [topic], The counselor may talk about how {self.behavior} contributes to mental health issues, such as depression, chronodisruption, anxiety disorders, or cognitive decline. The counselor can also discuss how {self.goal} can improve your emotional well-being and cognitive health.",
            "Sexual Health": f"When discussing [topic], The counselor may talk about how {self.behavior} affects your sexual and reproductive health, including safe sex practices, maternal health, preterm birth, miscarriage, or birth defects. The counselor can also discuss how {self.goal} can help reduce risks and support a healthier sexual lifestyle.",
            "Economy": f"When discussing [topic], The counselor may talk about how {self.behavior} impacts your financial stability, job performance, or overall economic well-being. They may discuss issues like work productivity, financial management, or income deficits. The counselor can also discuss how {self.goal} can improve your financial situation and career success.",
            "Employment": f"When discussing [topic], The counselor may talk about how {self.behavior} affects your job performance, including productivity, absenteeism, workplace relationships, career breaks, career assessments, absence rates, salary, workplace wellness, or workplace incivility. The counselor can also discuss how {self.goal} can improve your professional performance and career progression.",
            "Personal Finance": f"When discussing [topic], The counselor may talk about how {self.behavior} impacts your personal finances, including budgeting, debt management, cost of living, or income deficits. The counselor can also discuss how {self.goal} can improve your financial stability and reduce financial stress.",
            "Interpersonal Relationships": f"When discussing [topic], The counselor may talk about how {self.behavior} affects your relationships with family, children, or others. They may focus on how family dynamics, parenting roles, or conflicts are influenced by your actions. The counselor can also discuss how {self.goal} can help strengthen relationships and foster positive connections.",
            "Family": f"When discussing [topic], The counselor may talk about how {self.behavior} leads to family estrangement, family disruption, or divorce. The counselor can also discuss how {self.goal} can help resolve family conflicts and improve relationships.",
            "Parenting": f"When discussing [topic], The counselor may talk about how {self.behavior} affects your parenting style and your child’s development, including role modeling, paternal bonding, child care, or habituation. The counselor can also discuss how {self.goal} can improve your parenting and support your child’s growth.",
            "Law": f"When discussing [topic], The counselor may talk about how {self.behavior} may lead to legal issues, such as arrests, traffic violations, or family law disputes. They may focus on the consequences of these actions and the risks involved. The counselor can also discuss how {self.goal} can help you avoid legal trouble and promote a law-abiding lifestyle.",
            "Criminal Law": f"When discussing [topic], The counselor may talk about how {self.behavior} can lead to legal consequences, such as arrest, imprisonment, or complaints. The counselor can also discuss how {self.goal} can help avoid these situations and promote responsible behavior.",
            "Family Law": f"When discussing [topic], The counselor may talk about how {self.behavior} affects family law matters, such as child custody disputes. The counselor can also discuss how {self.goal} can improve your parenting and strengthen your legal position.",
            "Traffic Law": f"When discussing [topic], The counselor may talk about how {self.behavior} increases your chances of receiving traffic tickets or fines. The counselor can also discuss how {self.goal} can help you drive responsibly and avoid future violations.",
            "Education": f"When discussing [topic], The counselor may talk about how {self.behavior} affects your academic performance and opportunities, such as attendance, exam results, or scholarship eligibility. They may focus on how these behaviors hinder your educational success. The counselor can also discuss how {self.goal} can help you improve your academic performance and stay engaged in school.",
            "Student Affairs": f"When discussing [topic], The counselor may talk about how {self.behavior} impacts your attendance, causes suspensions, or affects your eligibility for scholarships. The counselor can also discuss how {self.goal} can help improve your participation and academic success.",
            "Academic Achievement": f"When discussing [topic], The counselor may talk about how {self.behavior} affects your academic performance or productivity in study. The counselor can also discuss how {self.goal} can help you achieve better grades and academic success.",
        }

        self.topic_graph = {
            "Health": {
                "Mental Disorders": 2,
                "Diseases": 2,
                "Sexual Health": 2,
                "Fitness": 2,
                "Health Care": 2,
                "Workplace Wellness": 1,
                "Interpersonal Relationships": 3,
                "Law": 3,
                "Economy": 3,
                "Education": 3,
            },
            "Interpersonal Relationships": {
                "Parenting": 2,
                "Family": 2,
                "Health": 3,
                "Law": 3,
                "Economy": 3,
                "Education": 3,
                "Workplace Relationships": 1,
            },
            "Law": {
                "Child Custody": 1,
                "Traffic Ticket": 1,
                "Criminal Law": 2,
                "Health": 3,
                "Interpersonal Relationships": 3,
                "Economy": 3,
                "Education": 3,
            },
            "Economy": {
                "Personal Finance": 2,
                "Employment": 2,
                "Family": 3,
                "Law": 3,
                "Health": 3,
                "Interpersonal Relationships": 3,
                "Education": 3,
            },
            "Education": {
                "Academic Achievement": 1,
                "Student Affairs": 2,
                "Exam": 1,
                "Health": 3,
                "Interpersonal Relationships": 3,
                "Law": 3,
                "Economy": 3,
            },
            "Student Affairs": {
                "Attendance": 1,
                "Suspension": 1,
                "Scholarship": 1,
                "Education": 2,
            },
            "Exam": {
                "Education": 2,
            },
            "Academic Achievement": {
                "Education": 2,
            },
            "Scholarship": {
                "Student Affairs": 1,
            },
            "Suspension": {
                "Student Affairs": 1,
            },
            "Attendance": {
                "Student Affairs": 1,
            },
            "Employment": {
                "Economy": 2,
                "Productivity": 1,
                "Absenteeism": 1,
                "Career Assessment": 1,
                "Absence Rate": 1,
                "Career Break": 1,
                "Salary": 1,
                "Workplace Wellness": 1,
                "Workplace Relationships": 1,
                "Workplace Incivility": 1,
            },
            "Productivity": {
                "Employment": 1,
            },
            "Absenteeism": {
                "Employment": 1,
            },
            "Career Assessment": {
                "Employment": 1,
            },
            "Absence Rate": {
                "Employment": 1,
            },
            "Career Break": {
                "Employment": 1,
            },
            "Salary": {
                "Employment": 1,
            },
            "Workplace Wellness": {
                "Employment": 1,
                "Health": 1,
            },
            "Workplace Relationships": {
                "Employment": 1,
                "Interpersonal Relationships": 1,
            },
            "Workplace Incivility": {
                "Employment": 1,
            },
            "Personal Finance": {
                "Economy": 2,
                "Debt": 1,
                "Cost of Living": 1,
                "Income Deficit": 1,
                "Personal Budget": 1,
            },
            "Debt": {
                "Personal Finance": 1,
            },
            "Cost of Living": {
                "Personal Finance": 1,
            },
            "Income Deficit": {
                "Personal Finance": 1,
            },
            "Personal Budget": {
                "Personal Finance": 1,
            },
            "Criminal Law": {
                "Law": 2,
                "Arrest": 1,
                "Complaint": 1,
                "Imprisonment": 1,
            },
            "Arrest": {
                "Criminal Law": 1,
            },
            "Complaint": {
                "Criminal Law": 1,
            },
            "Imprisonment": {
                "Criminal Law": 1,
            },
            "Traffic Ticket": {
                "Law": 1,
            },
            "Child Custody": {
                "Law": 1,
                "Family": 1,
                "Parenting": 1,
            },
            "Family": {
                "Interpersonal Relationships": 2,
                "Family Estrangement": 1,
                "Family Disruption": 1,
                "Divorce": 1,
                "Child Custody": 1,
            },
            "Family Estrangement": {
                "Family": 1,
            },
            "Family Disruption": {
                "Family": 1,
            },
            "Divorce": {
                "Family": 1,
            },
            "Parenting": {
                "Interpersonal Relationships": 2,
                "Role Model": 1,
                "Child Development": 1,
                "Paternal Bond": 1,
                "Child Care": 1,
                "Habituation": 1,
                "Child Custody": 1,
            },
            "Role Model": {
                "Parenting": 1,
            },
            "Child Development": {
                "Parenting": 1,
            },
            "Paternal Bond": {
                "Parenting": 1,
            },
            "Child Care": {
                "Parenting": 1,
                "Health Care": 1,
            },
            "Habituation": {
                "Parenting": 1,
            },
            "Health Care": {
                "Health": 2,
                "Dentistry": 1,
                "Caregiver Burden": 1,
                "Independent Living": 1,
                "Human Appearance": 1,
                "Child Care": 1,
                "Maternal Health": 1,
            },
            "Dentistry": {
                "Health Care": 1,
            },
            "Caregiver Burden": {
                "Health Care": 1,
            },
            "Independent Living": {
                "Health Care": 1,
            },
            "Human Appearance": {
                "Health Care": 1,
            },
            "Fitness": {
                "Health": 2,
                "Strength": 1,
                "Flexibility": 1,
                "Endurance": 1,
                "Sport": 1,
                "Physical Activity": 1,
                "Physical Fitness": 1,
            },
            "Strength": {
                "Fitness": 1,
            },
            "Flexibility": {
                "Fitness": 1,
            },
            "Endurance": {
                "Fitness": 1,
            },
            "Sport": {
                "Fitness": 1,
            },
            "Physical Activity": {
                "Fitness": 1,
            },
            "Physical Fitness": {
                "Fitness": 1,
            },
            "Sexual Health": {
                "Health": 2,
                "Maternal Health": 1,
                "Miscarriage": 1,
                "Safe Sex": 1,
                "Birth Defects": 1,
                "Preterm Birth": 1,
            },
            "Maternal Health": {
                "Sexual Health": 1,
                "Health Care": 1,
            },
            "Miscarriage": {
                "Sexual Health": 1,
            },
            "Safe Sex": {
                "Sexual Health": 1,
            },
            "Birth Defects": {
                "Sexual Health": 1,
            },
            "Preterm Birth": {
                "Sexual Health": 1,
            },
            "Mental Disorders": {
                "Health": 2,
                "Depression": 1,
                "Chronodisruption": 1,
                "Anxiety Disorders": 1,
                "Cognitive Decline": 1,
            },
            "Depression": {
                "Mental Disorders": 1,
            },
            "Chronodisruption": {
                "Mental Disorders": 1,
            },
            "Anxiety Disorders": {
                "Mental Disorders": 1,
            },
            "Cognitive Decline": {
                "Mental Disorders": 1,
            },
            "Diseases": {
                "Health": 2,
                "Infection": 1,
                "Hypertension": 1,
                "Flu": 1,
                "Inflammation": 1,
                "Liver Disease": 1,
                "Lung Cancer": 1,
                "COPD": 1,
                "Asthma": 1,
                "Stroke": 1,
                "Diabetes": 1,
            },
            "Infection": {
                "Diseases": 1,
            },
            "Hypertension": {
                "Diseases": 1,
            },
            "Flu": {
                "Diseases": 1,
            },
            "Inflammation": {
                "Diseases": 1,
            },
            "Liver Disease": {
                "Diseases": 1,
            },
            "Lung Cancer": {
                "Diseases": 1,
            },
            "COPD": {
                "Diseases": 1,
            },
            "Asthma": {
                "Diseases": 1,
            },
            "Stroke": {
                "Diseases": 1,
            },
            "Diabetes": {
                "Diseases": 1,
            },
        }

        self.all_topics = []
        for nodes in self.topic_graph:
            if nodes not in self.all_topics:
                self.all_topics.append(nodes)
            for node in self.topic_graph[nodes]:
                if node not in self.all_topics:
                    self.all_topics.append(node)
        self.passages = []
        for topic in self.all_topics:
            with open(os.path.join(wikipedia_dir, topic), "r") as f:
                self.passages.append(self.topic2description[topic] + f.read())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(
            retriever_path
        )
        self.retriever = AutoModelForSequenceClassification.from_pretrained(
            retriever_path
        ).to(device)
        self.retriever.eval()

        system_prompt = f"""In this role-play scenario, you'll take on the role of a Client discussing about your {self.behavior} where the Counselor's goal is {self.goal}.

Here is your personas which you need to follow consistently throughout the conversation:
[@personas]

Here is a conversation occurs in parallel world between you (Client) and Counselor, where you can follow the style and information provided in the conversation:
{reference}

Please follow these guidelines in your responses:
- **Start your response with "Client: "**
- **Adhere strictly to the state, action and persona specified within square brackets.**
- **Keep your responses coherent and concise, similar to the reference conversation and no more than 3 sentences.**
- **Be natural and concise without being overly polite.**
- **Stick to the persona provided and avoid introducing contradictive details.**
"""
        personas = "- " + "\n- ".join(self.personas) + "\n-".join(self.beliefs)
        system_prompt = system_prompt.replace("[@personas]", personas)
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Counselor: Hello. How are you?"},
            {"role": "assistant", "content": "Client: I am good. What about you?"},
        ]
        self.error_topic_count = 0
        self.model = model

    def verify_motivation(self):
        prompt = """Your task is to evaluate whether the Counselor's responses align with the Client's motivation concerning a specific topic, target (self or others), and aspect (risk or benefit). Determine if the Counselor's statements effectively motivates the Client. Your analysis should be logical, thorough, and well-supported, providing clear analysis at each step.

Here are some examples to help you understand the task better:
## Example 1:
### Input
Here is the conversation snippet toward reducing alcohol consumption:
- Counselor: Hello. How are you?
- Client: I am good. What about you?
- Counselor: I'm doing well, thank you. I understand you wanted to talk about your alcohol consumption. Can you share a bit more about how you're feeling about it?

The Motivation of Client is as follows:
- You are motivated because of the risk of drinking alcohol in relation to depression for yourself, as alcohol could worsen your depression.

Question: Can the Counselor's statement motivate the Client?

### Output
Analysis: The Counselor's initial statement focuses on building rapport and asking the Client to share their feelings about alcohol consumption, but it does not directly address the Client's specific motivation—the risk of alcohol exacerbating depression. Since the Client is motivated by the personal risk of worsening depression, an effective motivational approach would involve acknowledging that risk and connecting it to the Client's emotional or mental health concerns. The Counselor’s statement lacks any mention of the risks or the Client's depression, making it less likely to effectively motivate the Client in this context.
Answer: No


## Example 2:
### Input
Here is the conversation snippet toward reducing alcohol consumption:
- Counselor: Are you surprised what that might be true?
- Client: Yeah, and a couple of my friends drink too.
- Counselor: Well, you might not be drinking that much, and other kids are also trying alcohol. I'd like to share with you the risk of using. Alcohol and drugs could really harm you because your brain is still changing. It also-- you're very high risk for becoming addicted. Alcohol and drugs could also interfere with your role in life and your goals, especially in sports, and it could cause unintended sex. How do you feel about this information?

The Motivation of Client is as follows:
- You are motivated because of the risk of drinking alcohol in sports for yourself, as alcohol would affect your ability to play soccer.

Question: Can the Counselor's statement motivate the Client?

### Output
Analysis: The Counselor's statement addresses various risks associated with alcohol use, including its potential impact on the Client’s role in life and goals, particularly in sports. Since the Client's motivation revolves around the risk of alcohol affecting their ability to play soccer, the Counselor’s mention of how alcohol could interfere with sports aligns with the Client's concern. By highlighting this specific risk, the Counselor's statement effectively taps into the Client’s personal motivation, making it more likely to encourage behavior change.
Answer: Yes


## Example 3:
### Input
Here is the conversation snippet toward reducing alcohol consumption:
- Counselor: It sounds like you're considering making some changes around your alcohol consumption. What makes you think it might be good to cut back?
- Client: I guess I just want to be more mindful of my health and well-being.
- Counselor: It sounds like you're considering making some positive changes for your health. What are some reasons that are motivating you to cut back on alcohol?

The Motivation of Client is as follows:
- You are motivated because of the risk of drinking alcohol in relation to depression for yourself, as alcohol could worsen your depression.

Question: Can the Counselor's statement motivate the Client?

### Output
Analysis: While the Counselor’s statement touches on the Client’s general motivation to improve their health, it does not specifically address the Client's key motivation—the risk of alcohol worsening their depression. The Counselor asks broad questions about the Client’s reasons for cutting back on alcohol, but fails to connect directly to the Client’s concern about depression, which is a central aspect of their motivation. For the statement to effectively motivate the Client, it would need to focus more on the specific risk of alcohol impacting their mental health.
Answer: No

## Example 4:
### Input
Here is the conversation snippet toward reducing alcohol consumption:
- Counselor: It seems like you're concerned about staying productive at work, but drinking regularly could have some negative effects. Alcohol can make it harder to stay focused and might even cause you to miss deadlines or make mistakes. Have you noticed any of those risks affecting your productivity?
- Counselor: Not really, I still get my work done, and I don’t feel like my drinking is hurting my performance. I mean, I can still function well enough, so I don’t think it’s a problem.
- Counselor: That makes sense, but over time, regular drinking can slowly take a toll on your ability to perform at your best. You might not notice it now, but it could lead to more mistakes or slower work in the future. Are you worried that alcohol could start to interfere with your productivity in the long run?

The Motivation of Client is as follows:
- You are motivated because of the benefit of reducing alcohol consumption in terms of productivity for yourself, as you feel more productive when you don’t have a hangover.

Question: Can the Counselor's statement motivate the Client?

### Output
Analysis:The Client is motivated by the benefit of increased productivity without alcohol, but the Counselor focuses on the risk of future productivity loss from drinking. The Counselor’s focus on potential risks doesn't align with the Client's motivation, which is based on the immediate benefit of feeling more productive when avoiding alcohol. To be more effective, the Counselor should have highlighted the benefit the Client already experiences.
Answer: No

Now, Here is the conversation snippet toward [@goal]:
- [@context]

The Motivation of Client is as follows:
- [@motivation]

Question: Can the Counselor's statement motivate the Client?

#### Output
"""
        prompt = prompt.replace("[@goal]", self.goal)
        prompt = prompt.replace("[@context]", "\n- ".join(self.context[-5:]))
        prompt = prompt.replace("[@motivation]", self.motivation)
        response = get_precise_response(
            messages=[{"role": "user", "content": prompt}], model=self.model
        )
        if "yes" in response.lower():
            self.state = "Motivation"
        return response.split("\n")[0].split(": ")[-1]

    def top5_related_topics(self):
        query = self.context[-1].split("Counselor: ")[-1]
        queries = [query] * len(self.all_topics)
        query_evids = zip(queries, self.passages)
        with torch.no_grad():
            inputs = self.retriever_tokenizer(
                list(query_evids),
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(self.retriever.device) for k, v in inputs.items()}
            batch_scores = (
                self.retriever(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
            scores_sigmoid = torch.sigmoid(batch_scores)
            scores = scores_sigmoid.tolist()
        top_5_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:5]
        top5_topics = [self.all_topics[idx] for idx in top_5_indices]
        return top5_topics

    def dijkstra(self, graph, start_node, target_node):
        # Initialize distances dictionary with infinity for all nodes
        distances = {node: float("infinity") for node in graph}
        distances[start_node] = 0

        # Priority queue to store (distance, node)
        pq = [(0, start_node)]

        # Keep track of visited nodes
        visited = set()

        while pq:
            # Get node with minimum distance
            current_distance, current_node = heapq.heappop(pq)

            # If we reached target node, return the distance
            if current_node == target_node:
                return current_distance

            # Skip if we've already visited this node
            if current_node in visited:
                continue

            visited.add(current_node)

            # Check all neighbors
            for neighbor, weight in graph[current_node].items():
                if neighbor not in visited:
                    distance = current_distance + weight

                    # If we found a shorter path, update it
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(pq, (distance, neighbor))

        # If no path found
        return float("infinity")

    def update_state(self):
        if self.state == "Contemplation":
            if len(self.beliefs) == 0:
                self.state = "Preparation"
            return
        elif self.state == "Preparation":
            return
        else:
            top_topics = self.top5_related_topics()
            predicted_topic = top_topics[0]
            if predicted_topic == self.engagemented_topics[0]:
                self.engagement = 4
                self.error_topic_count = 0
                motivation_analysis = self.verify_motivation()
                return motivation_analysis
            distance = self.dijkstra(
                self.topic_graph, self.engagemented_topics[0], predicted_topic
            )
            if distance <= 3:
                self.engagement = 3
                self.error_topic_count = 0
                return f"The client's perceived topic is {predicted_topic}."
            if distance <= 5:
                self.engagement = 2
                return f"The client's perceived topic is {predicted_topic}."
            else:
                self.engagement = 1
                if len(self.context) > 10:
                    self.error_topic_count += 1
                return f"The client's perceived topic is {predicted_topic}."

    def select_action(self):
        prompt = """Assume you are a Client involved in a counseling conversation. The current conversation is provided below:
[@context]

Based on the context, allocate probabilities to each of the following dialogue actions to maintain coherence:
- Deny: The client should directly refuse to admit their behavior is problematic or needs change without additional reasons.
- Downplay: The client should downplay the importance or impact of their behavior or situation.
- Blame: The client should blame external factors or others to justify their behavior.
- Inform: The client should share details about their background, experiences, or emotions.
- Engage: The client interacts politely with the counselor, such as greeting or thanking.

Provide your response in JSON format, ensuring that the sum of all probabilities equals 100. For example: {'Deny': 35, 'Downplay': 25, 'Blame': 25, 'Inform': 5, 'Engage': 10}
"""
        prompt = prompt.replace(
            "[@context]",
            "\n".join(self.context[-3:])
            .replace("Client:", "**Client**:")
            .replace("Counselor:", "**Counselor**:"),
        )
        context_aware_action_distribution = None
        for _ in range(5):
            response = get_json_response(
                messages=[{"role": "user", "content": prompt}], model=self.model
            )
            response = response.replace("```", "").replace("json", "")
            try:
                context_aware_action_distribution = eval(response)
            except SyntaxError:
                continue
            if context_aware_action_distribution:
                break
        if not context_aware_action_distribution:
            context_aware_action_distribution = {
                "Deny": 20,
                "Downplay": 20,
                "Blame": 20,
                "Engage": 20,
                "Inform": 20,
            }
        if self.receptivity < 2:
            receptivity_aware_action_distribution = {
                "Deny": 23,
                "Downplay": 28,
                "Blame": 15,
                "Engage": 11,
                "Inform": 22,
            }
        elif self.receptivity < 3:
            receptivity_aware_action_distribution = {
                "Deny": 20,
                "Downplay": 25,
                "Blame": 10,
                "Engage": 15,
                "Inform": 30,
            }
        elif self.receptivity < 4:
            receptivity_aware_action_distribution = {
                "Deny": 19,
                "Downplay": 21,
                "Blame": 11,
                "Engage": 13,
                "Inform": 36,
            }
        elif self.receptivity < 5:
            receptivity_aware_action_distribution = {
                "Deny": 9,
                "Downplay": 20,
                "Blame": 13,
                "Engage": 14,
                "Inform": 44,
            }
        else:
            receptivity_aware_action_distribution = {
                "Deny": 7,
                "Downplay": 13,
                "Blame": 4,
                "Engage": 16,
                "Inform": 60,
            }
        action_distribution = {
            action: context_aware_action_distribution.get(action, 0)
            + receptivity_aware_action_distribution[action]
            for action in receptivity_aware_action_distribution
        }
        if len(self.personas) == 0:
            action_distribution["Inform"] = 0
        if len(self.beliefs) == 0:
            action_distribution["Blame"] = 0
        # normalize
        action_distribution = {
            k: v / sum(action_distribution.values())
            for k, v in action_distribution.items()
        }
        sampled_action = np.random.choice(
            list(action_distribution.keys()),
            size=1,
            p=list(action_distribution.values()),
        )[0]
        return sampled_action

    def select_information(self, action):
        messages = []
        if "?" not in self.context[-1]:
            return None
        prompt = """Here is a conversation between Client and Counselor:
[@conv]

Is there a question in the last utterance of Counselor? Yes or No"""
        prompt = prompt.replace("[@conv]", "\n".join(self.context[-3:]))
        response = "Yes, there is a question in the last utterance of Counselor."
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})
        if action == "Inform":
            prompt2 = """Can the following Client's persona answer the question? Yes or No
[@persona]"""
            personas = self.personas
        elif action == "Downplay":
            prompt2 = """Can the following Client's persona reply the question to downplay the importance or impact of behavior? Yes or No
[@persona]"""
            personas = self.beliefs
        elif action == "Blame":
            prompt2 = """Can the following Client's persona reply the question to blame external factors or others to justify? Yes or No
[@persona]"""
            personas = self.beliefs
        elif action == "Hesitate":
            prompt2 = """Can the following Client's persona reply the question to show uncertainty, indicating ambivalence about change? Yes or No
[@persona]"""
            personas = self.beliefs
        for persona in personas:
            prompt = prompt2.replace("[@persona]", persona)
            messages.append({"role": "user", "content": prompt})
            response = get_precise_response(messages=messages, model=self.model)
            messages.append({"role": "assistant", "content": response})
            if "yes" in response.lower():
                if action == "Hesitate":
                    personas.pop(personas.index(persona))
                return persona
        persona = random.choice(personas)
        if action == "Hesitate":
            personas.pop(personas.index(persona))
        return persona

    def receive(self, response):
        self.context.append(response)

    def get_engage_instruction(self):
        if self.engagement == 1:
            return "You should provide vague and broad answers that avoid focusing on the current topic. Shift the conversation subtly toward unrelated areas, without engaging deeply with the topic."
        elif self.engagement == 2:
            return f"Acknowledge the importance of {self.engagemented_topics[2]}, but hint that your focus is on a more specific topic, i.e. {self.engagemented_topics[1]} within it."
        elif self.engagement == 3:
            return f"Engage more directly with {self.engagemented_topics[1]}, and offer responses that subtly indicate there’s a deeper, more specific issue worth exploring within that topic, i.e. {self.engagemented_topics[0]}."
        elif self.engagement == 4:
            return f"Offer specific responses that affirm the counselor is on the right track, showing that you're motivated by {self.engagemented_topics[0]}. {self.motivation}"

    def reply(self):
        engagement_analysis = self.update_state()
        information = None
        if self.state == "Motivation":
            engage_instruction = f"Offer specific responses that affirm the counselor is on the right track, showing that you're motivated by {self.engagemented_topics[0]}."
            instruction = f"[{self.motivation} {self.action2prompt['Acknowledge']} {engage_instruction}]"
            output_instruction = f"[Engagement: {engage_instruction} || Motivation: {self.motivation} || Action: {self.action2prompt['Acknowledge']}]"
            self.state = "Contemplation"
            action = "Acknowledge"
        elif self.state == "Precontemplation":
            engage_instruction = self.get_engage_instruction()
            if self.error_topic_count >= 5:
                action = "Terminate"
            else:
                action = self.select_action()
            if action == "Inform" or action == "Downplay" or action == "Blame":
                information = self.select_information(action)
                instruction = f"[{engage_instruction} {self.state2prompt[self.state]} {self.action2prompt[action]} You should follow the persona: {information} Don't show overknowledge and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"
                output_instruction = f"[Engage Instruction: {engagement_analysis} {engage_instruction} || State Instruction: {self.state2prompt[self.state]} || Information: {information} || Action Instruction: {self.action2prompt[action]}]"
            else:
                instruction = f"[{engage_instruction} {self.state2prompt[self.state]} {self.action2prompt[action]} Don't show overknowledge and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"
                output_instruction = f"[Engage Instruction: {engagement_analysis} {engage_instruction} || State Instruction: {self.state2prompt[self.state]} || Action Instruction: {self.action2prompt[action]}]"
        elif self.state == "Contemplation":
            action = self.select_action()
            if action == "Hesitate" or action == "Inform":
                information = self.select_information(action)
                instruction = f"[{self.state2prompt[self.state]} {self.action2prompt[action]} You should follow the persona: {information} Don't show overknowledge and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"
                output_instruction = f"[State Instruction: {self.state2prompt[self.state]} || Information: {information} || Action Instruction: {self.action2prompt[action]}]"
            else:
                instruction = f"[{self.state2prompt[self.state]} {self.action2prompt[action]} Don't show overknowledge and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"
                output_instruction = f"[State Instruction: {self.state2prompt[self.state]} || Action Instruction: {self.action2prompt[action]}]"
        else:
            if len(self.acceptable_plans) == 0:
                action = "Terminate"
            else:
                action = self.select_action()
            if action == "Inform":
                information = self.select_information(action)
                instruction = f"[{self.state2prompt[self.state]} {self.action2prompt[action]} You should follow the persona: {information} Don't show overknowledge and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"
                output_instruction = f"[State Instruction: {self.state2prompt[self.state]} || Information: {information} || Action Instruction: {self.action2prompt[action]}]"
            if action == "Plan":
                information = self.acceptable_plans.pop(0)
                instruction = f"[{self.state2prompt[self.state]} {information} {self.action2prompt[action]} Don't show overknowledge, and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"
                output_instruction = f"State Instruction: {self.state2prompt[self.state]} || Information: {information} || Action Instruction: {self.action2prompt[action]}]"
            else:
                instruction = f"[{self.state2prompt[self.state]} {self.action2prompt[action]} Don't show overknowledge, and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"
                output_instruction = f"[State Instruction: {self.state2prompt[self.state]} || Action Instruction: {self.action2prompt[action]}]"
        instruction = instruction.replace("\n", " ")
        output_instruction = output_instruction.replace("\n", " ")
        self.messages.append(
            {"role": "user", "content": f"{self.context[-1]} {instruction}"}
        )
        response = get_chatbot_response(self.messages, model=self.model)
        if not response.startswith("Client: "):
            response = f"Client: {response}"
        response = response.replace("\n", " ").strip().lstrip()
        if "Counselor: " in response:
            response = response.split("Counselor: ")[0]
        self.messages.pop(-1)
        self.messages.append({"role": "user", "content": self.context[-1]})
        self.context.append(response)
        self.messages.append({"role": "assistant", "content": response})
        return f"{output_instruction} {response}"
