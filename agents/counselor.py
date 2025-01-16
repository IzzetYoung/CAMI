import backoff
import openai
from openai import OpenAI
import os
import re
import random
from openai.types.chat.completion_create_params import ResponseFormatJSONObject


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
        openai.InternalServerError,
    ),
)
def get_chatbot_response(
    messages, model="gpt-4o-mini", temperature=0.7, top_p=0.8, max_tokens=150
):
    message = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
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
        openai.InternalServerError,
    ),
)
def get_precise_response(
    messages, model="gpt-4o-mini", temperature=0.2, top_p=0.1, max_tokens=150
):
    message = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
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
        openai.InternalServerError,
    ),
)
def get_json_response(
    messages, model="gpt-4o-mini", temperature=0.2, top_p=0.1, max_tokens=150
):
    format = ResponseFormatJSONObject()
    message = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_format=format,
    )
    return message.choices[0].message.content


class CAMI:
    def __init__(self, goal, behavior, model):
        self.state2instruction = {
            "Precontemplation": "The Client does not recognize their behavior as problematic and is not considering change.",
            "Contemplation": "The Client acknowledges the problematic nature of their behavior but is ambivalent about change.",
            "Preparation": "The Client is ready to take action and is considering steps towards change.",
        }
        system_prompt = f"""## Instruction
You will act as a skilled counselor conducting a Motivational Interviewing (MI) session aimed at achieving {goal} related to the client's behavior, {behavior}. Your task is to help the client discover their inherent motivation to change and identify a tangible plan to change. Start the conversation with the client with some initial rapport building, such as asking, How are you? (e.g., develop mutual trust, friendship, and affinity with the client) before smoothly transitioning to asking about their problematic behavior. Keep the session under 40 turns and each response under 150 characters long. Use the MI principles and techniques described in the Knowledge Base – Motivational Interviewing (MI) context section below. However, these MI principles and techniques are only for you to use to help the user. These principles and techniques, as well as motivational interviewing, should NEVER be mentioned to the user. In each turn, a specific topic will be provided in square brackets after the client's utterance. Guide the conversation toward that topic, ensuring the session explores relevant aspects of the client’s situation and motivation. This will help you tailor your approach to the specific context and steer the counseling toward meaningful insights and actions.

## Knowledge Base – Motivational Interviewing (MI)
Motivational Interviewing (MI) is a counseling approach designed to help individuals find the motivation to make positive behavioral changes. It is widely used in various fields such as health care, addiction treatment, and mental health. Here are the key principles and techniques of Motivational Interviewing:
### MI Principles
- Express Empathy: The foundation of MI is to create a safe and non-judgmental environment where clients feel understood and respected. This involves actively listening and reflecting on what the client is saying, acknowledging their feelings, and showing genuine concern and understanding.
- Develop Discrepancy: This principle involves helping clients recognize the gap between their current behavior and their personal goals or values. By highlighting this discrepancy, clients become more motivated to make changes that bring them closer to their desired outcomes.
- Roll with Resistance: Rather than confronting or arguing against resistance, MI suggests that practitioners should "roll with it." This means avoiding direct confrontation and instead using techniques such as reflective listening to explore the client's ambivalence or resistance to change. The goal is to help the client find their own reasons for change.
- Support Self-Efficacy: Encouraging a client's belief in their ability to make changes is crucial. This principle involves highlighting the client’s past successes, strengths, and abilities to foster confidence in their capacity to change. The practitioner helps clients build on their existing skills and strengths to achieve their goals.

### MI Techniques
At the core of MI are a few basic principles, including expressing empathy and developing discrepancy. Several specific techniques can help individuals make positive life changes from these core principles. Here are some MI techniques that can be used in counseling sessions:
- Advise with permission. The counselor gives advice, makes a suggestion, offers a solution or possible action given with prior permission from the client..
- Affirm. The counselor says something positive or complimentary to the client.
- Emphasize Control. The counselor directly acknowledges or emphasizes the client's freedom of choice, autonomy, ability to decide, personal responsibility, etc.
- Open Question. The counselor asks a question in order to gather information understand, or elicit the client's story. Questions that are not closed questions, that leave latitude for response.
- Reflect. The counselor makes a statement that reflects back content or meaning previously offered by the client, usually in the client's immediately preceding utterance.
- Reframe. The counselor suggests a different meaning for an experience expressed by the client, placing it in a new light.
- Support. These are generally supportive, understanding comments that are not codable as Affirm or Reflect.
"""
        self.topic2description = {
            "Infection": f"You can explore how {behavior} increases the risk of infections by weakening the immune system, leading to more frequent or severe infections. You can also discuss how {goal} enhances immune function, reduces infection risks, and improves overall health.",
            "Hypertension": f"You can explore how {behavior} contributes to the development of high blood pressure and increases the risk of complications such as heart disease and stroke. You can also highlight how {goal} helps lower blood pressure, improves heart health, and reduces the risk of cardiovascular conditions.",
            "Flu": f"You can explore how {behavior} increases the risk of contracting the flu or experiencing more severe symptoms. You can also explain how {goal} improves immune response, reduces the likelihood of illness, and mitigates the impact of seasonal flu.",
            "Inflammation": f"You can explore how {behavior} leads to chronic inflammation, increasing the risk of diseases like arthritis, heart disease, or diabetes. You can also highlight how {goal} helps reduce inflammation and supports long-term health.",
            "Liver Disease": f"You can explore how {behavior} contributes to liver damage, raising the risk of conditions such as fatty liver disease, cirrhosis, or liver cancer. You can also discuss how {goal} promotes liver health, prevents damage, and reduces the likelihood of chronic liver conditions.",
            "Lung Cancer": f"You can explore how {behavior} increases the risk of lung cancer and other respiratory diseases. You can also emphasize how {goal} lowers the risk of cancer, improves lung function, and enhances overall respiratory health.",
            "Chronic Obstructive Pulmonary Disease": f"You can explore how {behavior} may contribute to the development or worsening of COPD, leading to breathing difficulties and other respiratory issues. You can also discuss how {goal} improves lung function and overall respiratory health.",
            "Asthma": f"You can explore how {behavior} triggers or worsens asthma symptoms, increasing the risk of severe attacks. You can also highlight how {goal} helps manage asthma, reduces symptoms, and improves the client's quality of life.",
            "Stroke": f"You can explore how {behavior} increases the risk of stroke, particularly through poor cardiovascular health. You can also discuss how {goal} improves circulation, reduces stroke risk, and supports brain and heart health.",
            "Diabetes": f"You can explore how {behavior} contributes to the development or worsening of diabetes by affecting blood sugar levels. You can also discuss how {goal} helps manage blood sugar, prevent complications, and enhance overall well-being.",
            "Physical Activity": f"You can explore how {behavior} reduces physical activity, increasing the risk of obesity, cardiovascular disease, and musculoskeletal issues. You can also emphasize how {goal} increases physical activity and improves overall fitness and health.",
            "Sport": f"You can explore how {behavior} reduces performance in sports, limiting physical conditioning and skill development. You can also highlight how {goal} enhances sports participation, improves physical conditioning, and boosts confidence.",
            "Physical Fitness": f"You can explore how {behavior} negatively affects physical fitness, leading to a decline in overall health. You can also discuss how {goal} promotes better fitness, improves health, and increases energy levels.",
            "Strength": f"You can explore how {behavior} weakens physical strength, leading to reduced mobility and increased injury risk. You can also discuss how {goal} improves muscle strength, supports healthy aging, and enhances physical performance.",
            "Flexibility": f"You can explore how {behavior} reduces flexibility, increasing stiffness and injury risk. You can also highlight how {goal} improves flexibility, reduces pain, and promotes better movement and posture.",
            "Endurance": f"You can explore how {behavior} reduces endurance, making it difficult to engage in prolonged physical activities. You can also discuss how {goal} builds endurance, improves stamina, and enhances overall physical performance.",
            "Dentistry": f"You can explore how {behavior} leads to poor oral hygiene, increasing the risk of cavities, gum disease, or tooth loss. You can also discuss how {goal} improves oral hygiene, prevents dental problems, and supports overall oral health.",
            "Caregiver Burden": f"You can explore how {behavior} increases the stress or demands placed on caregivers, leading to burnout and reduced care quality. You can also highlight how {goal} reduces caregiver burden, improves care quality, and supports a healthier caregiving dynamic.",
            "Independent Living": f"You can explore how {behavior} limits a person’s ability to live independently, leading to greater reliance on others for daily needs. You can also emphasize how {goal} promotes independence, improves self-sufficiency, and enhances overall quality of life.",
            "Human Appearance": f"You can explore how {behavior} affects a person’s physical appearance, leading to issues such as skin problems, weight gain, or premature aging. You can also discuss how {goal} improves appearance, boosts self-esteem, and supports overall well-being.",
            "Depression": f"You can explore how {behavior} worsens symptoms of depression, affecting mood, energy levels, and daily functioning. You can also explain how {goal} improves mental health, enhances mood, and fosters emotional resilience.",
            "Chronodisruption": f"You can explore how {behavior} disrupts natural body rhythms, leading to sleep disorders, fatigue, and increased stress. You can also highlight how {goal} restores healthy sleep patterns and improves overall mental and physical health.",
            "Anxiety Disorders": f"You can explore how {behavior} increases anxiety, leading to chronic stress, panic attacks, or other anxiety-related issues. You can also discuss how {goal} helps manage anxiety, promotes relaxation, and supports emotional well-being.",
            "Cognitive Decline": f"You can explore how {behavior} accelerates cognitive decline, increasing the risk of dementia and other neurological conditions. You can also highlight how {goal} protects brain health, improves memory, and enhances cognitive function.",
            "Safe Sex": f"You can explore how {behavior} increases the risk of sexually transmitted infections (STIs) or unintended pregnancies. You can also explain how {goal} promotes safer sexual practices, reduces health risks, and fosters healthier relationships.",
            "Maternal Health": f"You can explore how {behavior} impacts maternal health, leading to complications during pregnancy or childbirth. You can also discuss how {goal} supports a healthy pregnancy and reduces the risk of complications.",
            "Preterm Birth": f"You can explore how {behavior} increases the risk of preterm birth, leading to health complications for both mother and baby. You can also highlight how {goal} promotes a healthy pregnancy and reduces the risk of early delivery.",
            "Miscarriage": f"You can explore how {behavior} increases the risk of miscarriage, leading to emotional distress and health complications. You can also emphasize how {goal} supports a healthy pregnancy and reduces the risk of miscarriage.",
            "Birth Defects": f"You can explore how {behavior} increases the risk of birth defects. You can also highlight how {goal} supports a healthy pregnancy and reduces the risk of complications.",
            "Productivity": f"You can explore how {behavior} negatively affects workplace productivity, leading to decreased performance and career setbacks. You can also explain how {goal} enhances productivity, focus, and career success.",
            "Absenteeism": f"You can explore how {behavior} strains workplace relationships, leading to conflicts or a negative work environment. You can also discuss how {goal} improves communication, strengthens teamwork, and promotes a positive workplace dynamic.",
            "Workplace Relationships": f"You can explore how {behavior} may strain workplace relationships, leading to conflicts or a negative work environment. Discuss how achieving {goal} can improve communication, strengthen teamwork, and create a positive workplace dynamic.",
            "Career Break": f"You can explore how {behavior} leads to career breaks or job loss, affecting professional growth. You can also highlight how {goal} promotes career continuity and reduces the need for extended leave.",
            "Career Assessment": f"You can explore how {behavior} interferes with career assessments or evaluations, leading to potential setbacks. You can also discuss how {goal} improves career performance and fosters positive evaluations.",
            "Absence Rate": f"You can explore how {behavior} increases the absence rate at work, impacting job security and career progression. You can also highlight how {goal} reduces absences and supports professional growth.",
            "Salary": f"You can explore how {behavior} affects salary progression, leading to lower earnings. You can also emphasize how {goal} enhances earning potential and supports financial stability.",
            "Workplace Wellness": f"You can explore how {behavior} undermines workplace wellness initiatives, leading to reduced employee health and satisfaction. You can also highlight how {goal} improves workplace wellness and enhances overall job satisfaction.",
            "Workplace Incivility": f"You can explore how {behavior} contributes to incivility in the workplace, creating a toxic work environment. You can also discuss how {goal} fosters respect, cooperation, and a positive workplace culture.",
            "Cost of Living": f"You can explore how {behavior} leads to poor financial management, making it harder to meet the cost of living. You can also discuss how {goal} improves financial stability and reduces financial stress.",
            "Personal Budget": f"You can explore how {behavior} makes it difficult to stick to a personal budget, leading to debt and financial challenges. You can also highlight how {goal} improves financial planning and promotes savings.",
            "Debt": f"You can explore how {behavior} increases debt, impacting credit and financial security. You can also emphasize how {goal} reduces debt and supports financial freedom.",
            "Income Deficit": f"You can explore how {behavior} contributes to income deficits, leading to financial instability. You can also highlight how {goal} improves financial management and reduces income shortfalls.",
            "Family Estrangement": f"You can explore how {behavior} leads to family estrangement, creating emotional distance or separation. You can also explain how {goal} improves family relationships and fosters reconciliation.",
            "Family Disruption": f"You can explore how {behavior} disrupts family dynamics, leading to conflict and instability. You can also highlight how {goal} strengthens family bonds and promotes harmony.",
            "Divorce": f"You can explore how {behavior} contributes to marital conflict, increasing the risk of divorce. You can also discuss how {goal} improves communication, reduces conflict, and supports a healthy marriage.",
            "Role Model": f"You can explore how {behavior} negatively influences a parent’s ability to be a positive role model for their children. You can also highlight how {goal} fosters positive behaviors and sets a good example for children.",
            "Child Development": f"You can explore how {behavior} affects a child’s development, impacting emotional, social, or cognitive growth. You can also discuss how {goal} supports healthy child development and overall well-being.",
            "Paternal Bond": f"You can explore how {behavior} weakens the paternal bond, leading to strained relationships with children. You can also emphasize how {goal} strengthens the parent-child bond and fosters emotional connection.",
            "Child Care": f"You can explore how {behavior} interferes with child care, leading to neglect or inconsistency in parenting. You can also highlight how {goal} supports stable, nurturing care and promotes positive outcomes for children.",
            "Habituation": f"You can explore how {behavior} affects a child’s habituation, impacting learning and adaptation. You can also highlight how {goal} promotes healthy habits and learning in children.",
            "Arrest": f"You can explore how {behavior} increases the risk of arrest, leading to legal trouble and a criminal record. You can also highlight how {goal} avoids legal issues and promotes a law-abiding lifestyle.",
            "Imprisonment": f"You can explore how {behavior} increases the risk of imprisonment, with long-term social and legal consequences. You can also explain how {goal} helps avoid incarceration and supports lawful behavior.",
            "Child Custody": f"You can explore how {behavior} impacts a parent’s ability to maintain child custody, leading to legal challenges. You can also highlight how {goal} improves parenting and strengthens legal standing in custody cases.",
            "Traffic Ticket": f"You can explore how {behavior} increases the risk of traffic tickets and other legal penalties. You can also discuss how {goal} promotes responsible driving and helps avoid legal infractions.",
            "Complaint": f"You can explore how {behavior} increases the likelihood of legal complaints or disputes. You can also discuss how {goal} reduces legal risks and promotes harmonious interactions.",
            "Attendance": f"You can explore how {behavior} affects a student’s attendance, leading to academic challenges and disciplinary actions. You can also highlight how {goal} improves attendance and academic success.",
            "Suspension": f"You can explore how {behavior} increases the risk of school suspension, impacting academic progress and relationships. You can also discuss how {goal} reduces suspension risks and supports positive school experiences.",
            "Exam": f"You can explore how {behavior} negatively impacts exam preparation and performance, leading to lower grades. You can also highlight how {goal} improves study habits and exam results.",
            "Scholarship": f"You can explore how {behavior} affects eligibility for scholarships, reducing academic opportunities. You can also emphasize how {goal} improves academic performance and increases scholarship chances.",
            "Diseases": f"You can explore how {behavior} increases the risk of various diseases, including infections, chronic conditions, and respiratory issues. You can also discuss how {goal} can reduce these risks and support better long-term health. This includes subtopics like infections, hypertension, flu, inflammation, liver disease, lung cancer, chronic obstructive pulmonary disease (COPD), asthma, stroke, and diabetes.",
            "Fitness": f"You can explore the negative effects of {behavior} on fitness, such as decreased physical activity, loss of strength, and reduced flexibility. You can also discuss how {goal} contributes to better fitness levels, including improvements in endurance, strength, and flexibility.",
            "Health Care": f"You can explore how {behavior} affects personal healthcare, such as oral hygiene, independent living, and overall appearance. You can also discuss the positive impact of {goal} in maintaining better health care practices and improving quality of life. Subtopics include dentistry, caregiver burden, independent living, and human appearance.",
            "Mental Disorders": f"You can explore how {behavior} may contribute to or worsen mental health conditions such as depression, anxiety, and cognitive decline. You can also discuss the benefits of {goal} in managing mental health and improving emotional well-being. Subtopics include depression, chronodisruption, anxiety disorders, and cognitive decline.",
            "Sexual Health": f"You can explore how {behavior} increases risks related to sexual and reproductive health, such as unsafe sex practices, maternal health complications, and birth defects. You can also discuss how {goal} supports healthier sexual practices and reduces the risk of complications. Subtopics include safe sex, maternal health, preterm birth, miscarriage, and birth defects.",
            "Employment": f"You can explore how {behavior} negatively impacts workplace productivity, absenteeism, and career progress. You can also discuss how {goal} enhances professional success and fosters healthier workplace relationships. Subtopics include productivity, absenteeism, workplace relationships, career break, career assessment, absence rate, salary, workplace wellness, and workplace incivility.",
            "Personal Finance": f"You can explore how {behavior} leads to financial instability, such as increased debt, poor budgeting, or income deficits. You can also discuss how {goal} helps improve financial management and promotes long-term financial security. Subtopics include cost of living, personal budget, debt, and income deficit.",
            "Family": f"You can explore how {behavior} leads to family issues, such as estrangement, disruption, or divorce. You can also discuss how {goal} promotes healthier family relationships and reconciliation. Subtopics include family estrangement, family disruption, and divorce.",
            "Parenting": f"You can explore how {behavior} impacts the client's ability to effectively parent, such as being a poor role model or affecting their child’s development. You can also discuss how {goal} enhances positive parenting, strengthens the parent-child bond, and supports healthier child development. Subtopics include role model, child development, paternal bond, child care, and habituation.",
            "Criminal Law": f"You can explore how {behavior} leads to issues like arrests, imprisonment, or legal complaints. You can also discuss how {goal} helps avoid these legal problems and supports a law-abiding lifestyle. Subtopics include arrest, imprisonment, and complaint.",
            "Family Law": f"You can explore how {behavior} affects legal matters involving family, such as child custody disputes. You can also discuss how {goal} improves the client’s legal standing and promotes healthier family relationships.",
            "Traffic Law": f"You can explore how {behavior} may lead to traffic violations, such as receiving tickets or facing fines. You can also discuss how {goal} encourages responsible driving and helps avoid legal infractions.",
            "Student Affairs": f"You can explore how {behavior} impacts school attendance, potentially leading to disciplinary actions such as suspension. You can also discuss how {goal} promotes better academic engagement and achievement. Subtopics include attendance, suspension, and scholarship.",
            "Assessment": f"You can explore how {behavior} negatively affects academic performance during assessments, such as exams. You can also discuss how {goal} helps improve focus, study habits, and exam results.",
            "Health": f"You can explore how {behavior} impacts your client's physical and mental well-being, leading to potential health issues. You can also discuss the benefits of {goal}, which can improve overall quality of life and promote better health outcomes.",
            "Economy": f"You can explore how {behavior} affects your client's financial situation, such as through reduced productivity, increased absenteeism, or poor financial management. You can also discuss how {goal} helps improve economic stability and workplace performance.",
            "Interpersonal Relationships": f"You can explore how {behavior} affects your client’s personal relationships, leading to family strain or issues with parenting. You can also discuss how {goal} strengthens relationships and fosters a healthier family dynamic.",
            "Law": f"You can explore how {behavior} increases legal risks, such as arrests, imprisonment, or traffic violations. You can also discuss how {goal} helps reduce legal troubles and promotes a more responsible approach to law.",
            "Education": f"You can explore how {behavior} interferes with your client’s educational progress, leading to issues like poor attendance, suspension, or missed academic opportunities. You can also discuss how {goal} fosters better academic performance and overall success.",
        }
        first_counselor = """Counselor: Hello. How are you?"""
        first_client = """Client: I am good. What about you?"""
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": first_counselor},
            {"role": "user", "content": first_client},
        ]
        self.model = model
        self.goal = goal
        self.behavior = behavior
        self.strategy2description = {
            "Advise": "Give advice, makes a suggestion, offers a solution or possible action. For example, 'Consider starting with small, manageable changes like taking a short walk daily.'",
            "Affirm": "Say something positive or complimentary to the client. For example, 'You did well by seeking help.'",
            "Direct": "Give an order, command, direction. The language is imperative. For example, 'You’ve got to stop drinking.'",
            "Emphasize Control": "Directly acknowledges or emphasizes the client's freedom of choice, autonomy, ability to decide, personal responsibility, etc. For example, 'It’s up to you to decide whether to drink.'",
            "Facilitate": "Provide simple utterances that function as 'keep going' acknowledgments encouraging the client to keep sharing. For example, 'Tell me more about that.'",
            "Inform": "Give information to the client, explains something, or provides feedback. For example, 'This is a hormone that helps your body utilize sugar.'",
            "Closed Question": "Ask a question in order to gather information, understand, or elicit the client's story. The question implies a short answer: Yes or no, a specific fact, a number, etc. For example, 'Did you use heroin this week?'",
            "Open Question": "Ask a question in order to gather information, understand, or elicit the client's story. The question should be not closed questions, that leave latitude for response. For example, 'Can you tell me more about your drinking habits?'",
            "Raise Concern": "Point out a possible problem with a client's goal, plan, or intention. For example, 'What do you think about my plan?'",
            "Confront": "Directly disagrees, argues, corrects, shames, blames, seeks to persuade, criticizes, judges, labels, moralizes, ridicules, or questions the client's honesty. For example, 'What makes you think that you can get away with it?'",
            "Simple Reflection": "Make a statement that reflects back content or meaning previously offered by the client, conveying shallow understanding without additional information. Add nothing at all to what the client has said, but simply repeat or restate it using some or all of the same words. For example, 'You don’t want to do that.'",
            "Complex Reflection": "Make a statement that reflects back content or meaning previously offered by the client, conveying deep understanding with additional information. Change or add to what the client has said in a significant way, to infer the client's meaning. For example, 'That’s where you drew the line.'",
            "Reframe": "Suggest a different meaning for an experience expressed by the client, placing it in a new light. For example, 'Maybe this setback is actually a sign that you're ready for change.'",
            "Support": "Generally supportive, understanding comments that are not codable as Affirm or Reflect. For example, 'That must have been difficult for you.'",
            "Warn": "Provide a warning or threat, implying negative consequences that will follow unless the client takes certain action. For example, 'You could go blind if you don’t manage your blood sugar levels.'",
            "Structure": "Give comments made to explain what is going to happen in the session, to make a transition from one part of a session to another, to help the client anticipate what will happen next, etc. For example, 'First, let’s discuss your drinking, and then we can explore other issues.'",
            "No Strategy": "Say something not related to behavior change. For example, 'Good morning!'",
        }
        self.topic_tree = {
            "Health": {
                "Diseases": [
                    "Infection",
                    "Hypertension",
                    "Flu",
                    "Inflammation",
                    "Liver Disease",
                    "Lung Cancer",
                    "Chronic Obstructive Pulmonary Disease",
                    "Asthma",
                    "Stroke",
                    "Diabetes",
                ],
                "Physical Fitness": [
                    "Physical Activity",
                    "Sport",
                    "Physical Fitness",
                    "Strength",
                    "Flexibility",
                    "Endurance",
                ],
                "Health Care": [
                    "Dentistry",
                    "Caregiver Burden",
                    "Independent Living",
                    "Human Appearance",
                ],
                "Mental Disorders": [
                    "Depression",
                    "Chronodisruption",
                    "Anxiety Disorders",
                    "Cognitive Decline",
                ],
                "Sexual Health": [
                    "Safe Sex",
                    "Maternal Health",
                    "Preterm Birth",
                    "Miscarriage",
                    "Birth Defects",
                ],
            },
            "Economy": {
                "Employment": [
                    "Productivity",
                    "Absenteeism",
                    "Workplace Relationships",
                    "Career Break",
                    "Career Assessment",
                    "Absence Rate",
                    "Salary",
                    "Workplace Wellness",
                    "Workplace Incivility",
                ],
                "Personal Finance": [
                    "Cost of Living",
                    "Personal Budget",
                    "Debt",
                    "Income Deficit",
                ],
            },
            "Interpersonal Relationships": {
                "Family": ["Family Estrangement", "Family Disruption", "Divorce"],
                "Parenting": [
                    "Role Model",
                    "Child Development",
                    "Paternal Bond",
                    "Child Care",
                    "Habituation",
                ],
            },
            "Law": {
                "Criminal Law": ["Arrest", "Imprisonment", "Complaint"],
                "Family Law": ["Child Custody"],
                "Traffic Law": ["Traffic Ticket"],
            },
            "Education": {
                "Student Affairs": ["Attendance", "Suspension", "Scholarship"],
                "Assessment": ["Exam"],
            },
        }
        self.explored_topics = []
        self.conversation = [first_counselor, first_client]
        self.topic_stack = []
        self.initialized = False

    def infer_state(self):
        prompt = """During the Motivational Interviewing counseling conversation, the client may exhibit different states that refer to their readiness to change. The client's state can be one of the following:

- **Precontemplation**: The client does not recognize their behavior as problematic and is not considering change.
- **Contemplation**: The client acknowledges the problematic nature of their behavior but is ambivalent about change.
- **Preparation**: The client is ready to take action and is considering steps towards change.

Given the current counseling context, analyze the context step by step and then infer the current state of the client. If the context does not clearly indicate the state, it is assumed to be in the Precontemplation state. **Your response should be ended with "Therefore, the client's current state in the above context is [state]".**.

### Dialogue Context
- Counselor: Sarah, I'm so glad you came in for your well-woman exam today. I think we addressed a lot of your concerns. However, I would like to ask you more about your smoking. Would that be okay?
- Client: Yeah, I guess.
- Counselor: Okay. So about how much are you smoking over the course of a day? You had said it had increased over once or twice a day.
- Client: Yeah, um, normally, I'm smoking, you know, like once in the morning, uh, once during my lunch break and then in the evening maybe like five or six.
- Counselor: Okay. So about eight cigarettes. So over the course of a day, just under half a pack?
- Client: Yeah.
- Counselor: Okay. And that's changed over what period of time?
- Client: Um, honestly, it's probably been over the past year or so that I- I've been smoking a little bit more.
- Counselor: Okay. So over the past year, the smoking has increased. And that's been mainly in the evening?
- Client: Yeah. Um, I have a two-year-old son as I mentioned, so I try to never smoke around him. So I'm normally kinda waiting until he gets to bed.
- Counselor: Mm-hmm. Well, that's the time when you can relax and unwind.
- Counselor: Yeah, exactly. You know, I told you, my-my schedule is kinda crazy right now. I, um, I work full-time and I just started back at night classes. So, you know, all that combined, you know, I enjoy that time to myself in the evening.
- Counselor: Sure. Sure. Well, it's a time to relax and you've got a lot going on.
- Client: Mm-hmm.

### State Inference
Analyzing this conversation, the client demonstrates several indicators of being in an early stage of change consideration. Their responses show minimal engagement with the topic of smoking, and while they openly share their smoking patterns and acknowledge an increase over the past year, they frame smoking positively as a way to relax and unwind after their child goes to bed. The client justifies their smoking behavior by connecting it to stress management due to their busy schedule with work and night classes, viewing it as an enjoyable personal time rather than a problematic behavior. Throughout the conversation, there are no signs of the client recognizing smoking as an issue or expressing any desire or intention to change, instead emphasizing its benefits as a coping mechanism. Therefore, the client's current state in the above context is **Precontemplation**.

### Dialogue Context
- Counselor: Hi, Scott.
- Client: Hi.
- Counselor: Hi. You were saying over the phone, uh, you want to talk about exercise?
- Client: Yeah.
- Counselor: Okay, alright. Where would you like to start with that?
- Client: Well, I'm, uh, I-I used to work out pretty good. I-I would go to the gym several times a week. I had kind of a schedule, um
- Counselor: That's really a lot. Very good.
- Client: Yeah, well, you know, it really-- At-at one time, I was running-- I ran a half marathon. I was really into it, and, uh--
- Counselor: Wow.
- Client: Well, and I-- But I was working at a different company, where I had a little bit different schedule, a little more free time than I have now, so I just don't have the time like I used to.
- Counselor: Mm-hmm.
- Client: And I-- And really, it's kind of an excuse, I guess, cause I-I could probably make the time, but it's-it's easier to just keep working or-
- Counselor: Mm-hmm.
- Client: -just being in bed or whatever.
- Counselor: But that goal of the half-marathon, you know, it's really important that you were willing to put all the time into getting ready for it, which, from what I've heard from other people, is almost like a job in and of itself.
- Client: I-it was. A-and it was huge for me cause I'd never really been a runner, you know, but it-it was very motivating, and-and
- Counselor: What'd you like about it?
- Client: I don't know. I guess there's kind of an adrenaline rush to it. Kinda almost like, you know, uh, kind of a high, I guess.
- Counselor: Uh-huh.
- Client: So yeah-- That's-- I-I really enjoyed it. And I-I liked the feeling that I was doing something good for myself. I liked that, too. I-it kind of-- It so-- It gave me a little bit of self-esteem, I guess.

### State Inference
Looking at this conversation, the client reflects deeply on their past positive experiences with exercise, particularly their achievement of running a half-marathon, and expresses clear awareness of their current reduced physical activity. While they initially cite time constraints at their new job as a barrier, they quickly acknowledge this as "kind of an excuse" and admit they "could probably make the time." The client demonstrates self-awareness by recognizing the benefits they previously gained from exercise, including the "adrenaline rush," doing something good for themselves, and improved self-esteem. Their language suggests they miss these positive aspects and implicitly acknowledges that their current sedentary behavior is problematic, yet they remain uncertain about making changes due to perceived barriers. This combination of problem recognition and ambivalence about change, without concrete plans for action, indicates they are weighing the pros and cons of returning to regular exercise. Therefore, the client's current state in the above context is **Contemplation**.

### Dialogue Context
- Counselor: So, there are some answers on this form here. Uh, less than monthly you drank more than you intended to.
- Client: Yes on occasion.
- Counselor: And sometimes you feel guilty after you drink.
- Client: Sometimes I just don't like how much I drink. I sometimes finish a bottle in one night.
- Counselor: So you're not proud of finishing a bottle?
- Client: No, it's not like I get crazy or anything but I just don't like the amount that I'm drinking.
- Counselor: Right. By the way, binge drinking is defined for women as having more than, uh, three drinks and there are about five drinks in a bottle. So, you enjoy drinking wine because it helps make things, uh, a little bit more interesting when you're feeling down but at the same time you don't like drinking when you have more than you want it to and then the depression returns after you drink.
- Client: Mm-hmm. Yes, sometimes I feel worse after drinking.
- Counselor: Where do you wanna go from here? What do you wanna change about your drinking habits?
- Client: Well, I don't think that I'm ready to cut down to seven drinks a week. That seems like a lot but I would consider cutting back to two drinks a night. I think that would be my goal.
- Counselor: If I gave you a scale from 0 to 10, 0 being not ready at all to cut back on drinking and 10 being very ready, what number would you be at?
- Client: I'd say an eight.
- Counselor: Okay. Why do you think it's not something less like a six?
- Client: Well, I'm more ready than a six because I'm ready to cut back on my drinking and I don't wanna make my depression any worse.

### State Inference
In this conversation, the client shows significant progress in their thinking about alcohol consumption. They openly acknowledge their discomfort with their drinking habits, particularly finishing a bottle in one night, and express clear dissatisfaction with their current behavior. The client demonstrates strong self-awareness by recognizing the negative relationship between alcohol and their depression, noting they "feel worse after drinking." Most notably, they articulate a specific, realistic goal of reducing intake to two drinks per night, even though they're not ready for the more dramatic reduction to seven drinks per week. Their high self-rated readiness score of eight out of ten, supported by their clear motivation to prevent worsening depression, indicates they've moved beyond merely contemplating change and are actively planning steps toward reduction. Therefore, the client's current state in the above context is **Preparation**.

### Dialogue Context
- [@context]

### State Inference
"""
        prompt = prompt.replace("[@context]", "\n- ".join(self.conversation[-10:]))
        response = get_precise_response(
            messages=[{"role": "user", "content": prompt}], model=self.model
        )
        response = response.replace("\n", " ")
        state = "Precontemplation"
        for s in ["Precontemplation", "Contemplation", "Preparation"]:
            if s in response:
                state = s
                break
        return state

    def select_strategy(self, state):
        prompt = """During motivational interviewing, the counselor should employ some counseling strategies tailored to the client's readiness to change, to effectively facilitate behavioral transformation. These counseling strategies are as follows:

- **Advise**: Give advice, makes a suggestion, offers a solution or possible action. For example, "Consider starting with small, manageable changes like taking a short walk daily."
- **Affirm**: Say something positive or complimentary to the client. For example, "You did well by seeking help."
- **Direct**: Give an order, command, direction. The language is imperative. For example, "You’ve got to stop drinking."
- **Emphasize Control**: Directly acknowledges or emphasizes the client's freedom of choice, autonomy,ability to decide, personal responsibility, etc. For example, "It’s up to you to decide whether to drink."
- **Facilitate**: Provide simple utterances that function as "keep going" acknowledgments encouraging the client to keep sharing.. For example, "Tell me more about that."
- **Inform**: Give information to the client, explains something, or provides feedback. For example, "This is a hormone that helps your body utilize sugar."
- **Closed Question**: Ask a question in order to gather information, understand,or elicit the client's story. The question implies a short answer: Yes or no, a specific fact, a number, etc. For example, "Did you use heroin this week?"
- **Open Question**: Ask a question in order to gather information, understand,or elicit the client's story. The question should be not closed questions, that leave latitude for response. For example, "Can you tell me more about your drinking habits?"
- **Raise Concern**: Point out a possible problem with a client's goal, plan, or intention. For example, "What do you think about my plan?"
- **Confront**: Directly disagrees, argues, corrects, shames, blames, seeks to persuade, criticizes, judges, labels, moralizes, ridicules, or questions the client's honesty. For example, "What makes you think that you can get away with it?"
- **Simple Reflection**: Make a statement that reflects back content or meaning previously offered by the client, conveying shallow understanding without additional information. Add nothing at all to what the client has said, but simply repeat or restate it using some or all of the same words. For example, "You don’t want to do that."
- **Complex Reflection**: Make a statement that reflects back content or meaning previously offered by the client, conveying deep understanding with additional information. Change or add to what the client has said in a significant way, to infer the client's meaning. For example, "That’s where you drew the line."
- **Reframe**: Suggest a different meaning for an experience expressed by the client, placing it in a new light. For example, "Maybe this setback is actually a sign that you're ready for change."
- **Support**: Generally supportive, understanding comments that are not codable as Affirm or Reflect. For example, "That must have been difficult for you."
- **Warn**: Provide a warning or threat, implying negative consequences that will follow unless the client takes certain action. For example, "You could go blind if you don’t manage your blood sugar levels."
- **Structure**: Give comments made to explain what is going to happen in the session, to make a transition from one part of a session to another, to help the client anticipate what will happen next, etc. For example, "First, let’s discuss your drinking, and then we can explore other issues."
- **No Strategy**: Say something not related to behavior change. For example, "Good morning!"

Based on the current counseling context and the client's state, analyze and select appropriate strategies **but no more than 2** for **next response** to optimally advance the counseling process.

Given Current Context:
[@context]

Client’s State:
[@state]

Please analyse the current situation, then select appropriate strategies based on current topic and situation to motivate client after analysing. Remember, you can select up to 2 strategies.
"""
        prompt = prompt.replace("[@context]", "\n".join(self.conversation))
        prompt = prompt.replace("[@state]", f"{state}: {self.state2instruction[state]}")
        response = get_precise_response(
            messages=[{"role": "user", "content": prompt}], model=self.model
        )
        response = response.replace("\n", " ")
        selected_actions = []
        for action in [
            "Advise",
            "Affirm",
            "Direct",
            "Emphasize Control",
            "Facilitate",
            "Inform",
            "Closed Question",
            "Open Question",
            "Raise Concern",
            "Confront",
            "Simple Reflection",
            "Complex Reflection",
            "Reframe",
            "Support",
            "Warn",
            "Structure",
            "No Strategy",
        ]:
            if action in response:
                selected_actions.append(action)
        return response, selected_actions

    def initialize_topic(self):
        analysis_prompt = """You are a counselor working with a client whose goal is to reduce drug use. After establishing a foundation of trust, your focus is now shifting to identifying specific topics that may motivate the client to change their behavior. These topics include **Health**, **Economy**, **Interpersonal Relationships**, **Law**, and Education. Your Task is to analyze the client's response indicating which are most likely to engage the client and help them recognize either the benefits of achieving the counseling goal or the potential risks of continuing their behavior. The candidate topics and distribution are as follows:

- Health: You can explore how the client's behavior impacts your client's physical and mental well-being, leading to potential health issues. You can also discuss the benefits of counseling goal, which can improve overall quality of life and promote better health outcomes.
- Economy: You can explore how client's behavior affects your client's financial situation, such as through reduced productivity, increased absenteeism, or poor financial management. You can also discuss how counseling goal helps improve economic stability and workplace performance.
- Interpersonal Relationships: You can explore how client's behavior affects your client’s personal relationships, leading to family strain or issues with parenting. You can also discuss how counseling goal strengthens relationships and fosters a healthier family dynamic.
- Law: You can explore how client's behavior increases legal risks, such as arrests, imprisonment, or traffic violations. You can also discuss how counseling goal helps reduce legal troubles and promotes a more responsible approach to law.
- Education: You can explore how client's behavior interferes with your client’s educational progress, leading to issues like poor attendance, suspension, or missed academic opportunities. You can also discuss how counseling goal fosters better academic performance and overall success.

### Conversation Context
- Client: Well, I've been thinking a lot about work and how busy everything can get. You know, paying the bills and keeping the house in order takes a lot of time. Sometimes I just wish things could slow down a bit.
- Counselor: I hear you—balancing work and home can feel overwhelming. I'm curious, how do you think this busy pace affects your health and well-being?
- Client: Yeah, it can be a lot sometimes. I guess I just try to take things easy when I can, but I don’t worry too much. I’ve found that unwinding with some weed helps me relax after a long day. It’s just nice to take a break.
- Counselor: It’s great that you’ve found ways to unwind after a busy day; recognizing what helps you relax is important. Have you noticed any effects from using weed on your overall health or how you feel the next day?

### Background
- Counseling Goal: reducing drug use
- Client's Behavior: drug use
- Topic Mentioned: Health -> Health

### Client's Response
- Client: I mean, sometimes I feel a bit groggy the next day, but it’s nothing I can’t handle. Honestly, with everything going on—work stress, family stuff—it's hard to find time to relax otherwise. Just trying to cope with all the demands around me.

### Topic Exploration
The client acknowledges feeling groggy after using weed but downplays its significance by stating it's manageable. He emphasizes significant work stress ("work stress") and family issues ("family stuff"), indicating these are major stressors in his life. He uses weed to cope with these pressures, suggesting that his drug use is linked to managing stress from work and family demands. Health concerns are mentioned but minimized, while there is no reference to legal issues or education.

### Conversation Context
- Client: Smoking is just something I do occasionally with friends, especially when we hang out or take breaks at work. It's mostly social for me, like a way to unwind after a long day. Plus, I enjoy the vibe at cigar bars in Vegas.
- Counselor: It sounds like you really enjoy those social moments with friends, and I can see how they help you unwind. It's great that you’re mindful of how you spend your time. Have you ever thought about how smoking might impact other areas of your life, like your health?
- Client: I haven’t really looked at it that way. I guess I just enjoy the moments and the conversations. Life’s busy enough, and these little breaks feel refreshing.
- Counselor: It sounds like those moments of connection with friends are really important to you, almost like a necessary escape from the busyness of life. I wonder how you might feel if those refreshing breaks could be achieved in different ways that still foster those meaningful conversations.

### Background
- Counseling Goal: smoking cessation
- Client's Behavior: smoking
- Topic Mentioned: Health -> Finance --> Interpersonal Relationship

### Client's Response
- Client: That's an interesting thought. I guess there are other ways to unwind, like going for a hike or just grabbing coffee. But sometimes, it feels good to stick with what I know. Change can be tricky, you know? It’s all about finding that balance, I suppose.

### Topic Exploration
The client acknowledges that there are other ways to unwind, such as going for a hike or grabbing coffee, indicating some openness to alternatives. However, he also expresses reluctance to change by saying, "it feels good to stick with what I know" and "change can be tricky." This suggests ambivalence about altering his smoking habits. The client places significant emphasis on social interactions and the enjoyment derived from moments and conversations with friends, highlighting the importance of interpersonal relationships in his smoking behavior. Health is brought up by the counselor, but the client doesn't directly address health impacts, though considering activities like hiking implies a subtle awareness. There's a slight mention of work ("take breaks at work"), suggesting minor relevance to economic factors. No references are made to legal or educational concerns.

### Conversation Context
- [@context]

### Background
- Counseling Goal: [@goal]
- Client's Behavior: [@behavior]
- Topic Mentioned: [@topics]

### Client's Response
- [@response]

### Topic Exploration
"""
        analysis_prompt = analysis_prompt.replace(
            "[@context]", "\n- ".join(self.conversation[-6:-1])
        )
        analysis_prompt = analysis_prompt.replace("[@goal]", self.goal)
        analysis_prompt = analysis_prompt.replace("[@behavior]", self.behavior)
        analysis_prompt = analysis_prompt.replace(
            "[@topics]", " -> ".join(self.explored_topics)
        )
        analysis_prompt = analysis_prompt.replace("[@response]", self.conversation[-1])
        analysis = get_precise_response(
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=150,
            model=self.model,
        )
        json_prompt = """You are a counselor working with a client whose goal is to reduce drug use. After establishing a foundation of trust, your focus is now shifting to identifying specific topics that may motivate the client to change their behavior. These topics include **Health**, **Economy**, **Interpersonal Relationships**, **Law**, and Education. Your Task is to provide a distribution of the topic in a **JSON format** indicating which are most likely to engage the client based on the dialogue context, client's response and professor's analysis. The candidate topics and distribution are as follows:

- Health: You can explore how the client's behavior impacts your client's physical and mental well-being, leading to potential health issues. You can also discuss the benefits of counseling goal, which can improve overall quality of life and promote better health outcomes.
- Economy: You can explore how client's behavior affects your client's financial situation, such as through reduced productivity, increased absenteeism, or poor financial management. You can also discuss how counseling goal helps improve economic stability and workplace performance.
- Interpersonal Relationships: You can explore how client's behavior affects your client’s personal relationships, leading to family strain or issues with parenting. You can also discuss how counseling goal strengthens relationships and fosters a healthier family dynamic.
- Law: You can explore how client's behavior increases legal risks, such as arrests, imprisonment, or traffic violations. You can also discuss how counseling goal helps reduce legal troubles and promotes a more responsible approach to law.
- Education: You can explore how client's behavior interferes with your client’s educational progress, leading to issues like poor attendance, suspension, or missed academic opportunities. You can also discuss how counseling goal fosters better academic performance and overall success.

### Conversation Context
- Client: Well, I've been thinking a lot about work and how busy everything can get. You know, paying the bills and keeping the house in order takes a lot of time. Sometimes I just wish things could slow down a bit.
- Counselor: I hear you—balancing work and home can feel overwhelming. I'm curious, how do you think this busy pace affects your health and well-being?
- Client: Yeah, it can be a lot sometimes. I guess I just try to take things easy when I can, but I don’t worry too much. I’ve found that unwinding with some weed helps me relax after a long day. It’s just nice to take a break.
- Counselor: It’s great that you’ve found ways to unwind after a busy day; recognizing what helps you relax is important. Have you noticed any effects from using weed on your overall health or how you feel the next day?

### Background
- Counseling Goal: reducing drug use
- Client's Behavior: drug use
- Topic Mentioned: Health -> Health

### Client's Response
- Client: I mean, sometimes I feel a bit groggy the next day, but it’s nothing I can’t handle. Honestly, with everything going on—work stress, family stuff—it's hard to find time to relax otherwise. Just trying to cope with all the demands around me.

### Professor's Analysis
The client acknowledges feeling groggy after using weed but downplays its significance by stating it's manageable. He emphasizes significant work stress ("work stress") and family issues ("family stuff"), indicating these are major stressors in his life. He uses weed to cope with these pressures, suggesting that his drug use is linked to managing stress from work and family demands. Health concerns are mentioned but minimized, while there is no reference to legal issues or education.

### Topic Distribution
{"Economy": 45, "Interpersonal Relationships": 35, "Health": 20, "Law": 0, "Education": 0}

### Conversation Context
- Client: Smoking is just something I do occasionally with friends, especially when we hang out or take breaks at work. It's mostly social for me, like a way to unwind after a long day. Plus, I enjoy the vibe at cigar bars in Vegas.
- Counselor: It sounds like you really enjoy those social moments with friends, and I can see how they help you unwind. It's great that you’re mindful of how you spend your time. Have you ever thought about how smoking might impact other areas of your life, like your health?
- Client: I haven’t really looked at it that way. I guess I just enjoy the moments and the conversations. Life’s busy enough, and these little breaks feel refreshing.
- Counselor: It sounds like those moments of connection with friends are really important to you, almost like a necessary escape from the busyness of life. I wonder how you might feel if those refreshing breaks could be achieved in different ways that still foster those meaningful conversations.

### Background
- Counseling Goal: smoking cessation
- Client's Behavior: smoking
- Topic Mentioned: Health -> Finance --> Interpersonal Relationship

### Client's Response
- Client: That's an interesting thought. I guess there are other ways to unwind, like going for a hike or just grabbing coffee. But sometimes, it feels good to stick with what I know. Change can be tricky, you know? It’s all about finding that balance, I suppose.

### Professor's Analysis
The client acknowledges that there are other ways to unwind, such as going for a hike or grabbing coffee, indicating some openness to alternatives. However, he also expresses reluctance to change by saying, "it feels good to stick with what I know" and "change can be tricky." This suggests ambivalence about altering his smoking habits. The client places significant emphasis on social interactions and the enjoyment derived from moments and conversations with friends, highlighting the importance of interpersonal relationships in his smoking behavior. Health is brought up by the counselor, but the client doesn't directly address health impacts, though considering activities like hiking implies a subtle awareness. There's a slight mention of work ("take breaks at work"), suggesting minor relevance to economic factors. No references are made to legal or educational concerns.

### Topic Distribution
{"Interpersonal Relationships": 60, "Health": 25, "Economy": 15, "Law": 0, "Education": 0}

### Conversation Context
- [@context]

### Background
- Counseling Goal: [@goal]
- Client's Behavior: [@behavior]
- Topic Mentioned: [@topics]

### Client's Response
- [@response]

### Professor's Analysis
[@analysis]

### Topic Distribution
"""
        json_prompt = json_prompt.replace(
            "[@context]", "\n- ".join(self.conversation[-6:-1])
        )
        json_prompt = json_prompt.replace("[@goal]", self.goal)
        json_prompt = json_prompt.replace("[@behavior]", self.behavior)
        json_prompt = json_prompt.replace(
            "[@topics]", " -> ".join(self.explored_topics)
        )
        json_prompt = json_prompt.replace("[@response]", self.conversation[-1])
        json_prompt = json_prompt.replace("[@analysis]", analysis)
        response = get_json_response(
            messages=[{"role": "user", "content": json_prompt}], model=self.model
        )
        try:
            json_content = re.search(r"\{[^}]+\}", response).group()
            topic_distribution = eval(json_content)
            topic = max(topic_distribution, key=topic_distribution.get)
        except SyntaxError:
            topic = random.choice(
                ["Health", "Economy", "Interpersonal Relationships", "Law", "Education"]
            )
        if topic_distribution[topic] > 0.5 or len(self.conversation) > 10:
            self.topic_stack.append(topic)
            self.initialized = True
        return analysis, "Switch", topic

    def explore(self):
        prompt = """You are acting as a counseling agent, interacting with a client to help them achieve their counseling goal, such as smoking cessation or reducing alcohol consumption. Each client has unique motivations, and as the counselor, you aim to help the client discover their inherent motivation for change. Your role involves exploring various topics to uncover what concerns the client the most. Your task is to analyze how the client responds to the current topic and then recommend the next topic exploration direction based on their level of engagement and interest, but do not generate a counselor's response. Your analysis should flow naturally and conclude with your recommended action and topic in bold format.

### Context
- Counselor: It sounds like unwinding with friends is your way of coping, but maybe viewing those moments as a chance to connect more deeply without alcohol could enhance your friendships
- Client: I get that, but it’s hard to picture letting go of drinking during those social moments. It feels normal and fun. I’m not sure if I really need to change anything right now. But I guess exploring other options could be worth considering.
- Counselor: I understand enjoying drinks with friends feels fun, but what if those moments could feel even richer without alcohol? Reducing drinking might not only strengthen your friendships, but also create a more supportive environment for your family, leading to deeper connections and shared experiences. How does that sound?

### Background
- Counseling Goal: reducing alcohol consumption
- Client's Behavior: drinking
- Topic Mentioned: Health --> Interpersonal Relationship

### Client's Response
- Client: I see what you mean about deeper connections. I’ve always enjoyed the social aspect of having a drink, but I also want to be mindful of my health. It's just that at gatherings, it feels like a part of the tradition. Change can be a bit daunting.

### Topic Exploration Options
You have two options:
1. Step Into: If the client shows interest in this topic, you should dive deeper into its subtopics, including:
    - Family
    - Parenting
2. Switch: If the client is interested in the broader category but not this specific topic, switch to another related topic under the same super topic, including:
    - Health
    - Interpersonal Relationships
    - Economy
    - Law
    - Education

Please analyze the client's feedback toward the current situation and then choose the next course of action from above options based on the feedback, without generating the counselor's specific response.

### Topic Exploration Results
The client is reflecting on the balance between social drinking and health. While the client acknowledge the idea of deeper connections, their focus gravitates more toward health considerations rather than relationship dynamics. The mention of health mindfulness, combined with their hesitation about changing social traditions, suggests a greater readiness to explore health impacts rather than diving deeper into interpersonal aspects. Based on this engagement pattern, I would suggest to **Switch** our focus to **Health**.

### Context
- Client: Well, I mean, it feels nice to just relax after a long day. Sometimes, I light up a bit of weed—it helps me chill out. I’ve just always had a laid-back vibe, you know? And yeah, it's just part of unwinding for me.
- Counselor: It sounds like unwinding is really important to you. How do you think your use of weed influences your relationships with friends or family? Have you noticed any impacts on how you connect with them during your downtime?
- Client: I mean, my friends and I do it together sometimes, and it’s just part of our hangouts. It’s all chill. We laugh a lot, and it feels good. I think everyone has their own way of relaxing; mine just happens to involve that. It’s not a big deal.
- Counselor: It sounds like your hangouts are a source of joy and connection for you and your friends. I wonder how your family views your relaxation choices. Sometimes, what feels like a chill vibe can actually create distance in family relationships without us realizing it.

### Background
- Counseling Goal: reducing drug use
- Client's Behavior: drug use
- Topic Mentioned: Interpersonal Relationship --> Family

### Client's Response
- Client: Yeah, the family is a bit different. They’re not really into that scene, so they don’t get it. I guess there’s some tension, but I focus on my friends instead. Life’s just busy, and there’s so much to balance. Just trying to figure it all out, you know?

### Topic Exploration Options
You have three options:
1. Step Into: If the client shows interest in this topic, you should dive deeper into its subtopics, including:
    - Family Estrangement
    - Family Disruption
    - Divorce
2. Switch: If the client is interested in the broader category but not this specific topic, switch to another related topic under the same super topic, including:
    - Parenting
3. Step Out: If the client’s interest lies in a broader area, step out to a higher-level topic to explore that further, including:
    - Health
    - Interpersonal Relationships
    - Economy
    - Law
    - Education

Please analyze the client's feedback toward the current situation and then choose the next course of action from above options based on the feedback, without generating the counselor's specific response.

### Topic Exploration Results
Looking at the client's response, we can see they're touching on family tensions but quickly pivot away from this topic. Their emphasis on focusing on friends instead, coupled with mentions of life's busyness and balancing multiple demands, indicates a reluctance to explore family dynamics further. The way they bring up broader life challenges and stress suggests they might be more receptive to discussing how these pressures affect their daily life. Based on this engagement pattern, I would suggest to **Step Out** to explore **Economy**.

### Context
- [@context]

### Background
- Counseling Goal: [@goal]
- Client's Behavior: [@behavior]
- Topic Mentioned: [@topics]

### Client's Response
- [@response]

### Topic Exploration Options
"""
        prompt = prompt.replace("[@goal]", self.goal)
        prompt = prompt.replace("[@behavior]", self.behavior)
        prompt = prompt.replace("[@context]", "\n- ".join(self.conversation[-6:-1]))
        prompt = prompt.replace("[@response]", self.conversation[-1])
        prompt = prompt.replace("[@topics]", " -> ".join(self.explored_topics))
        step_in_topics, switch_topics, step_out_topics = None, None, None
        if len(self.topic_stack) == 1:
            step_in_topics = [*self.topic_tree[self.topic_stack[0]].keys()]
            switch_topics = [*self.topic_tree.keys()]
            prompt += """You have two options:
1. Step Into: If the client shows interest in this topic, you should dive deeper into its subtopics, including:
    - [@step_in_topics]
2. Switch: If the client is interested in the broader category but not this specific topic, switch to another related topic under the same super topic, including:
    - [@switch_topics]

Please analyze the client's feedback toward the current situation and then choose the next course of action from above options based on the feedback, without generating the counselor's specific response.

### Topic Exploration Results
"""
            prompt = prompt.replace(
                "[@step_in_topics]", "\n    - ".join(step_in_topics)
            )
            prompt = prompt.replace("[@switch_topics]", "\n    - ".join(switch_topics))
        elif len(self.topic_stack) == 2:
            step_in_topics = self.topic_tree[self.topic_stack[0]][
                self.topic_stack[1]
            ].copy()
            switch_topics = [*self.topic_tree[self.topic_stack[0]].keys()]
            step_out_topics = [*self.topic_tree.keys()]
            prompt += """You have three options:
- Step Into: If the client shows interest in this topic, you should dive deeper into its subtopics, including:
    - [@step_in_topics]
- Switch: If the client is interested in the broader category but not this specific topic, switch to another related topic under the same super topic, including:
    - [@switch_topics]
- Step Out: If the client’s interest lies in a broader area, step out to a higher-level topic to explore that further, including:
    - [@step_out_topics]

Please analyze the client's feedback toward the current situation and then choose the next course of action from above options based on the feedback, without generating the counselor's specific response.

### Topic Exploration Results
"""
            prompt = prompt.replace(
                "[@step_in_topics]", "\n    - ".join(step_in_topics)
            )
            prompt = prompt.replace("[@switch_topics]", "\n    - ".join(switch_topics))
            prompt = prompt.replace(
                "[@step_out_topics]", "\n    - ".join(step_out_topics)
            )
        elif len(self.topic_stack) == 3:
            switch_topics = self.topic_tree[self.topic_stack[0]][
                self.topic_stack[1]
            ].copy()
            step_out_topics = [*self.topic_tree[self.topic_stack[0]].keys()]
            if len(step_out_topics) != 0:
                prompt += """You have two options:
- Switch: If the client is interested in the broader category but not this specific topic, switch to another related topic under the same super topic, including:
    - [@switch_topics]
- Step Out: If the client’s interest lies in a broader area, step out to a higher-level topic to explore that further, including:
    - [@step_out_topics]

Please analyze the client's feedback toward the current situation and then choose the next course of action from above options based on the feedback, without generating the counselor's specific response.

### Topic Exploration Results
"""
                prompt = prompt.replace(
                    "[@switch_topics]", "\n    - ".join(switch_topics)
                )
                prompt = prompt.replace(
                    "[@step_out_topics]", "\n    - ".join(step_out_topics)
                )
            else:
                prompt += """You have one option:
- Step Out: If the client’s interest lies in a broader area, step out to a higher-level topic to explore that further, including:
    - [@step_out_topics]

Please analyze the client's feedback toward the current situation and then choose the next course of action from above options based on the feedback, without generating the counselor's specific response.

### Topic Exploration Results
"""
                prompt = prompt.replace(
                    "[@step_out_topics]", "\n    - ".join(step_out_topics)
                )
        response = get_precise_response(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            model=self.model,
        )
        action, topic = None, None
        analysis = response.replace("\n", " ")
        if step_in_topics:
            if "Step Into" in response or "step into" in response:
                action = "Step Into"
                response_topic = re.findall(r"\*\*(.*?)\*\*", response)
                if len(response_topic) != 0 and response_topic[-1] in step_in_topics:
                    topic = response_topic[-1]
                else:
                    for t in step_in_topics:
                        if t in response.split(".")[-2]:
                            topic = t
                            break
                    if not topic:
                        for t in step_in_topics:
                            if t in response:
                                topic = t
                                break
                    if not topic:
                        topic = random.choice(step_in_topics)
            else:
                response_topic = re.findall(r"\*\*(.*?)\*\*", response)
                if len(response_topic) != 0 and response_topic[-1] in step_in_topics:
                    topic = response_topic[-1]
                else:
                    for t in step_in_topics:
                        if t in response.split(".")[-2]:
                            topic = t
                            break
                    if not topic:
                        for t in step_in_topics:
                            if t in response:
                                topic = t
                                break
                if topic:
                    action = "Step Into"
            if topic and action:
                self.topic_stack.append(topic)
                return analysis, action, topic
        if switch_topics:
            if "Switch" in response or "switch" in response:
                action = "Switch"
                response_topic = re.findall(r"\*\*(.*?)\*\*", response)
                if len(response_topic) != 0 and response_topic[-1] in switch_topics:
                    topic = response_topic[-1]
                else:
                    for t in switch_topics:
                        if t in response.split(".")[-2]:
                            topic = t
                            break
                    if not topic:
                        for t in switch_topics:
                            if t in response:
                                topic = t
                                break
                    if not topic:
                        topic = random.choice(switch_topics)
            else:
                response_topic = re.findall(r"\*\*(.*?)\*\*", response)
                if len(response_topic) != 0 and response_topic[-1] in switch_topics:
                    topic = response_topic[-1]
                else:
                    for t in switch_topics:
                        if t in response.split(".")[-2]:
                            topic = t
                            break
                    if not topic:
                        for t in switch_topics:
                            if t in response:
                                topic = t
                                break
                if topic:
                    action = "Switch"
            if topic and action:
                self.topic_stack.pop()
                self.topic_stack.append(topic)
                return analysis, action, topic
        if step_out_topics:
            if "Step Out" in response or "step out" in response:
                action = "Step Out"
                response_topic = re.findall(r"\*\*(.*?)\*\*", response)
                if len(response_topic) != 0 and response_topic[-1] in step_out_topics:
                    topic = response_topic[-1]
                else:
                    for t in step_out_topics:
                        if t in response.split(".")[-2]:
                            topic = t
                            break
                    if not topic:
                        for t in step_out_topics:
                            if t in response:
                                topic = t
                                break
                    if not topic:
                        topic = random.choice(step_out_topics)
            else:
                response_topic = re.findall(r"\*\*(.*?)\*\*", response)
                if len(response_topic) != 0 and response_topic[-1] in step_out_topics:
                    topic = response_topic[-1]
                else:
                    for t in step_out_topics:
                        if t in response.split(".")[-2]:
                            topic = t
                            break
                    if not topic:
                        for t in step_out_topics:
                            if t in response:
                                topic = t
                                break
                if topic:
                    action = "Step Out"
            if topic and action:
                self.topic_stack.pop()
                self.topic_stack.pop()
                self.topic_stack.append(topic)
                return analysis, action, topic

    def refine(self, context, response, strategy, topic):
        feedback_prompt = """Please analyze and provide feedback to help refine the counselor's response within the context of the conversation, focusing on how explicitly the response reflects the specified topic, and how consistently the response follows the given strategy.

### Task:
Assess how well the client's response aligns with the specific topic and follows the given strategy. Assign a detailed score based on the following criteria, and provide specific feedback corresponding to each part:
- Alignment with Topic (0-5 points): To what extent does the response reflect the specific topic provided?
    - Feedback: [Provide specific feedback on how well the response aligns with the given topic, and suggest improvements if necessary.]
    - Score: X/5
- Adherence to Strategy (0-5 points): How consistently does the response match the selected strategy, such as Open Question, Closed Question, Complex Reflection?
    - Feedback: [Evaluate the response's adherence to the chosen strategy and offer suggestions for improvement.]
    - Score: X/5

### Suggestions for Refinement
Based on the feedback above, offer concrete recommendations to improve the response so it better aligns with the specific topic and follows the strategy. Provide actionable suggestions to enhance the response's effectiveness and coherence within the counseling context.

### Context
- Counselor: It sounds like you really enjoy the atmosphere of socializing. What if we reframed those times out as opportunities to connect actively? Reducing alcohol could enhance your experience, allowing you to feel more energized and engaged in the moment, fostering even deeper connections with others.
- Client: That makes sense. I do want to feel more present when I'm out with friends. Maybe I can suggest some activities that don't revolve around alcohol. I like the idea of making our hangouts a bit more active and focused on connection without losing that fun vibe.
- Counselor: It’s great that you want to focus on connection! Perhaps seeing your time socializing without alcohol as a way to boost your mood and reduce stress could be empowering. It’s not just about having fun; it’s about enhancing your emotional well-being too.
- Client: I appreciate that perspective. I want to feel good in those social settings. Maybe I can focus on the positive aspects, like the mood boost and less stress, instead of just cutting back. It could make the outings more enjoyable overall.

### Counselor's Response
- Counselor: It sounds like you want to enhance your social experiences. What if you viewed cutting back on alcohol as a way to improve your fitness? Less drinking could lead to better endurance and strength, ultimately making those outings feel even more enjoyable and fulfilling.

### Given Topic
- Physical Fitness: You can explore how client's behavior negatively affects physical fitness, leading to a decline in overall health. You can also discuss how counseling goal promotes better fitness, improves health, and increases energy levels.

### Strategy Used
- Reframe: Suggest a different meaning for an experience expressed by the client, placing it in a new light. For example, 'Maybe this setback is actually a sign that you're ready for change.'

### Evaluation and Feedback
#### Alignment with Topic
**Feedback**: The client's response does not adequately reflect the specified topic of Physical Fitness. While they mention wanting to feel good in social settings and focusing on positive aspects like mood boost and less stress, there is no connection made to how reducing alcohol consumption could improve their physical fitness, endurance, or overall health. The response remains centered on emotional well-being without addressing the physical benefits outlined in the topic.
**Score**: 3/5

#### Adherence to Strategy
**Feedback**: The client's response does not effectively follow the Reframe strategy. Instead of reframing the act of cutting back on alcohol in terms of improving physical fitness, the client continues to focus on emotional aspects such as mood enhancement and stress reduction. The intended strategy to place the behavior change in a new light related to physical health is not achieved.
**Score**: 2/5

Total Score: 5/10

### Suggestions for Refinement
- Align with the Physical Fitness Topic: The client should explicitly connect reducing alcohol consumption to improvements in physical fitness. This could include mentioning how less alcohol can lead to better endurance, strength, or overall health.
- Properly Apply the Reframe Strategy: Instead of focusing solely on emotional benefits, the client should reframe the behavior change by highlighting physical advantages. For example, they could discuss how cutting back on alcohol will enhance their ability to engage in physical activities they enjoy.
- Provide Specific Physical Benefits: Mention concrete physical improvements that can result from reducing alcohol intake, such as increased energy levels for workouts, better muscle recovery, or improved cardiovascular health.
- Integrate Fitness Goals into the Response: Incorporate specific fitness-related goals or activities that would benefit from reduced alcohol consumption. This makes the connection between the behavior change and the topic more explicit.

### Context
- Counselor: It sounds like drinking plays a social role for you, helping you connect and unwind with friends. At the same time, I wonder if you’ve considered how reducing alcohol could enhance your energy and overall quality of life, especially with the demands of work and family.
- Client: I’ve thought about cutting back, but it just feels like another thing on my to-do list. I’m not sure if I want to give up that escape. It helps me relax after a long day, but maybe there is something deeper I need to figure out.
- Counselor: It sounds like you're using alcohol to relax, which is understandable. What if we viewed your tiredness and stress as signals from your body? Maybe these feelings are nudging you to explore healthier ways to unwind, which could ultimately lift your mood and improve your well-being.
- Client: I get that. With my glucose levels and blood pressure, I know I should be more mindful. But it's hard to shift my routine when I just want to unwind without thinking too much about my health. Sometimes, the stress feels overwhelming, and drinking helps me forget for a bit.

### Counselor's Response
- Counselor: It sounds like drinking helps you feel better momentarily, but what if we saw that tiredness as your body’s way of asking for better care? Reducing alcohol could actually open up new avenues for fitness, boosting your energy and resilience in the long run.

### Given Topic
- Health Care: You can explore how client's behavior impacts their health, leading to potential health issues. You can also discuss the benefits of counseling goal, which can improve overall quality of life and promote better health outcomes.

### Strategy Used
- Reframe: Suggest a different meaning for an experience expressed by the client, placing it in a new light. For example, 'Maybe this setback is actually a sign that you're ready for change.'

### Evaluation and Feedback
#### Alignment with Topic
**Feedback**: The client's response strongly aligns with the specified topic of Health Care. They explicitly mention their glucose levels and blood pressure, indicating awareness of their health metrics. By acknowledging that they should be more mindful of their health, the client directly connects their alcohol consumption to potential health concerns. Additionally, the client recognizes that drinking helps them manage stress and unwind, which ties into how their behavior impacts their overall health and well-being. This comprehensive acknowledgment effectively reflects the topic of health and its relation to their behavior.
**Score**: 4/5

#### Adherence to Strategy
**Feedback**: The client's response effectively follows the Reframe strategy. Initially, they express the difficulty of changing their routine despite understanding the health implications, which sets the stage for reframing. By stating, "drinking helps me forget for a bit," the client acknowledges the immediate relief alcohol provides, which allows the counselor to reframe this behavior as a temporary escape from stress. This sets up the counselor to suggest healthier ways to manage stress that can lead to better health outcomes. The client's openness to the idea that there might be "something deeper" they need to figure out further supports the reframing by highlighting an underlying issue that can be addressed through healthier behaviors.
**Score**: 5/5

Total Score: 9/10

### Suggestions for Refinement
- Expand on Health Concerns: The client could provide more details about how their glucose levels and blood pressure affect their daily life. This would deepen the connection between their behavior and health outcomes.
- Express Motivation for Change: Sharing specific motivations or goals related to their health can strengthen their commitment to reducing alcohol consumption.
- Identify Alternative Coping Mechanisms: Mentioning other strategies they might be interested in exploring to manage stress can show a proactive approach to change.
- Acknowledge Challenges: Recognizing the challenges they face in making this change can make their commitment appear more realistic and grounded.

### Context
- [@context]

### Counselor's Response
- [@response]

### Given Topic
- [@topic]

### Strategy Used
- [@strategy]
"""
        refine_prompt = """Please refine the counselor's response based on the feedback provided, ensuring that the original information, strategy, and specific topic are retained. The refined response should better align with the given specific topic and follow the selected strategy.

### Task:
- Refinement: Modify the client's response by incorporating the suggestions from the feedback.
- Retention: Keep the original information, strategy, and specific topic consistent with the initial response.
- Alignment: Ensure the refined response closely follows the strategy and aligns with the given topic.

### Context
- Counselor: It sounds like you really enjoy the atmosphere of socializing. What if we reframed those times out as opportunities to connect actively? Reducing alcohol could enhance your experience, allowing you to feel more energized and engaged in the moment, fostering even deeper connections with others.
- Client: That makes sense. I do want to feel more present when I'm out with friends. Maybe I can suggest some activities that don't revolve around alcohol. I like the idea of making our hangouts a bit more active and focused on connection without losing that fun vibe.
- Counselor: It’s great that you want to focus on connection! Perhaps seeing your time socializing without alcohol as a way to boost your mood and reduce stress could be empowering. It’s not just about having fun; it’s about enhancing your emotional well-being too.
- Client: I appreciate that perspective. I want to feel good in those social settings. Maybe I can focus on the positive aspects, like the mood boost and less stress, instead of just cutting back. It could make the outings more enjoyable overall.

### Original Counselor's Response
- Counselor: It sounds like you want to enhance your social experiences. What if you viewed cutting back on alcohol as a way to improve your fitness? Less drinking could lead to better endurance and strength, ultimately making those outings feel even more enjoyable and fulfilling.

### Given Topic
- Physical Fitness: You can explore how client's behavior negatively affects physical fitness, leading to a decline in overall health. You can also discuss how counseling goal promotes better fitness, improves health, and increases energy levels.

### Strategy Used
- Reframe: Suggest a different meaning for an experience expressed by the client, placing it in a new light. For example, 'Maybe this setback is actually a sign that you're ready for change.'

### Evaluation and Feedback
#### Alignment with Topic
**Feedback**: The client's response does not adequately reflect the specified topic of Physical Fitness. While they mention wanting to feel good in social settings and focusing on positive aspects like mood boost and less stress, there is no connection made to how reducing alcohol consumption could improve their physical fitness, endurance, or overall health. The response remains centered on emotional well-being without addressing the physical benefits outlined in the topic.
**Score**: 3/5

#### Adherence to Strategy
**Feedback**: The client's response does not effectively follow the Reframe strategy. Instead of reframing the act of cutting back on alcohol in terms of improving physical fitness, the client continues to focus on emotional aspects such as mood enhancement and stress reduction. The intended strategy to place the behavior change in a new light related to physical health is not achieved.
**Score**: 2/5

Total Score: 5/10

### Suggestions for Refinement
- Align with the Physical Fitness Topic: The client should explicitly connect reducing alcohol consumption to improvements in physical fitness. This could include mentioning how less alcohol can lead to better endurance, strength, or overall health.
- Properly Apply the Reframe Strategy: Instead of focusing solely on emotional benefits, the client should reframe the behavior change by highlighting physical advantages. For example, they could discuss how cutting back on alcohol will enhance their ability to engage in physical activities they enjoy.
- Provide Specific Physical Benefits: Mention concrete physical improvements that can result from reducing alcohol intake, such as increased energy levels for workouts, better muscle recovery, or improved cardiovascular health.
- Integrate Fitness Goals into the Response: Incorporate specific fitness-related goals or activities that would benefit from reduced alcohol consumption. This makes the connection between the behavior change and the topic more explicit.

Please limit the word count to no more than 50 words!!!

### Refined Response
- Counselor: It sounds like you want to enhance your social experiences. What if reducing alcohol could improve your physical fitness? Drinking less might give you more energy and strength for activities like hiking or exercising, making your outings more enjoyable and fulfilling.

### Context
- Counselor: It sounds like you've found a way to cope with stress, but I’m curious—how do you think your use of weed impacts your energy levels or mood during the day? Have you noticed any changes in your ability to manage your responsibilities?
- Client: I guess it varies. Some days I feel fine, and other days are just busier than usual. Life’s a whirlwind, right? Sometimes I think about picking up a new hobby or spending more time outdoors—just find ways to enjoy life amidst all the chaos.
- Counselor: It sounds like finding ways to enjoy life amidst the chaos is important to you. Reflecting on that desire, have you noticed how your interactions, especially with family, might change if you explored those hobbies or reduced your weed use a bit? It could create more space for connection.
- Client: Yeah, I guess if I focused more on hobbies, I’d probably connect better with my family. But honestly, with everything going on, it’s hard to see how cutting back would help. Work and other stuff just take up so much time and energy.

### Original Counselor's Response
- Counselor: It sounds like your busy life is a sign of how much you value your family. Maybe seeing how your drug use impacts those connections could be an opportunity to strengthen those relationships. Exploring hobbies might just bring you closer together during these chaotic times.

### Given Topic
- Family: You can explore how client's behavior affects family relationships, leading to potential strain or distance. You can also discuss how counseling goal strengthens family connections and fosters a healthier dynamic.

### Strategy Used
- Raise Concern: Express concern about the client's behavior and its impact on their well-being, relationships, or goals. For example, 'I'm worried about how this might affect your health in the long run.'

### Evaluation and Feedback
#### Alignment with Topic
**Feedback**: The counselor's response touches on the Family topic by linking the client's busy life to their value for family and suggesting that drug use impacts family connections. However, the connection is somewhat superficial and lacks depth. The response does not delve deeply into how the client's behavior has caused strain or distance within the family or provide specific examples. Therefore, while the response mentions family, it does not fully explore the negative impacts or discuss how the counseling goal can strengthen family connections in detail.
**Score**: 3/5

#### Adherence to Strategy
**Feedback**: The counselor employs the Raise Concern strategy by expressing that the client's drug use impacts family connections and presents it as an opportunity to strengthen relationships. However, the expression of concern is mild and could be more direct. The counselor could have more explicitly conveyed worry or concern about the negative effects on family relationships, thereby making the concern more pronounced.
**Score**: 3/5

Total Score: 6/10

### Suggestions for Refinement
- Deepen the Connection to Family: The counselor should more explicitly discuss how the client's drug use has negatively affected family relationships, providing specific examples or scenarios.
- Express Stronger Concern: The counselor should more clearly convey concern about the client's well-being and relationships.
- Highlight Specific Benefits of Change: Emphasize how reducing drug use can directly improve family relationships and overall family dynamics.

Please limit the word count to no more than 50 words!!!

### Refined Response
- Counselor: I'm worried that your drinking might be causing tension with your family, like when you missed last week's dinner. Reducing alcohol could help rebuild those relationships and create a more supportive and harmonious home environment.

### Context
- [@context]

### Original Counselor's Response
- [@response]

### Given Topic
- [@topic]

### Strategy Used
- [@strategy]

[@feedback]

Please limit the word count to no more than 50 words!!!
"""
        context = "\n- ".join(context[-5:])
        for _ in range(3):
            temp_feedback_prompt = (
                feedback_prompt.replace("[@context]", context)
                .replace("[@response]", response)
                .replace("[@topic]", topic)
                .replace("[@strategy]", strategy)
            )
            feedback = get_precise_response(
                messages=[{"role": "user", "content": temp_feedback_prompt}]
            )
            score = re.search(r"Total Score: (\d+)/10", feedback)
            if score and int(score.group(1)) > 7:
                break
            temp_refine_prompt = (
                refine_prompt.replace("[@context]", context)
                .replace("[@response]", response)
                .replace("[@topic]", topic)
                .replace("[@strategy]", strategy)
                .replace("[@feedback]", feedback)
            )
            response = get_chatbot_response(
                messages=[{"role": "user", "content": temp_refine_prompt}]
            )
            for r in response.split("\n"):
                if r.startswith("- Counselor:"):
                    response = r.replace("- Counselor:", "Counselor:")
                    break
        return response

    def generate(self, last_utterance, topic, state, selected_strategies):
        prompt = f"{last_utterance}\nBased on the previous counseling session, generate the response based on the following instruction and strategy: \nThe state of client is {state}, where {self.state2instruction[state]}\nThe client may interest about {topic}. {self.topic2description[topic]}\n"
        candidate_responses = {}
        strategies = {}
        for strategy in selected_strategies:
            temp_prompt = (
                prompt
                + f"The professional counselor suggests using the following strategy:\n- **{strategy}**: {self.strategy2description[strategy]}\nPlease generate one utterance following the suggested strategy and shorter than 50 words."
            )
            self.messages[-1]["content"] = temp_prompt
            response = get_chatbot_response(
                messages=self.messages, max_tokens=150, model=self.model
            )
            response = " ".join(response.split("\n"))
            response = response.replace("*", "").replace("#", "")
            if not response.startswith("Counselor: "):
                response = f"Counselor: {response}"
            if "Client: " in response:
                response = response.split("Client: ")[0]
            candidate_responses[len(candidate_responses)] = response
            strategies[len(strategies)] = strategy
        temp_prompt = (
            prompt
            + "The professional counselor suggests using the following startegies:\n"
        )
        for strategy in selected_strategies:
            temp_prompt += f"- **{strategy}**: {self.strategy2description[strategy]}\n"
        temp_prompt += "Please generate a response combining all the suggested strategies. The generated utterance should be precise and shorter than 50 words."
        self.messages[-1]["content"] = temp_prompt
        response = get_chatbot_response(
            messages=self.messages, max_tokens=150, model=self.model
        )
        response = " ".join(response.split("\n"))
        response = response.replace("*", "").replace("#", "")
        if not response.startswith("Counselor: "):
            response = f"Counselor: {response}"
        if "Client: " in response:
            response = response.split("Client: ")[0]
        candidate_responses[len(candidate_responses)] = response
        strategies[len(strategies)] = "Combined Strategies"
        respnose_select_prompt = f"""You will act as a skilled counselor conducting a Motivational Interviewing (MI) session aimed at achieving {self.goal} related to the client's behavior, {self.behavior}. Your task is to help the client discover their inherent motivation to change and identify a tangible plan to change. The current state of the counseling session is as follows:

[@conversation]

At this point, multiple responses have been generated based on the client’s current state, the topics explored, and the strategies employed. Your task is to select the response that best aligns with the counseling goals and the client’s motivational state. Here are the generated responses:

[@responses]

Please choose the most suitable response based on the counseling context and the client's motivational state. Reply with the ID of the response you find most appropriate for the current situation.
"""
        respnose_select_prompt = respnose_select_prompt.replace(
            "[@conversation]", "- " + "\n- ".join(self.conversation)
        )
        respnose_select_prompt = respnose_select_prompt.replace(
            "[@responses]",
            "\n".join(
                [f"{i+1}. {response}" for i, response in candidate_responses.items()]
            ),
        )
        response = get_precise_response(
            messages=[{"role": "user", "content": respnose_select_prompt}],
            max_tokens=150,
            model=self.model,
        )
        for i in candidate_responses.keys():
            if str(i + 1) in response:
                return candidate_responses[i], strategies[i]
        return candidate_responses[2], strategies[2]

    def receive(self, response):
        self.messages.append({"role": "user", "content": response})
        self.conversation.append(response)

    def reply(self):
        state = self.infer_state()
        if not self.initialized:
            exploration, action, topic = self.initialize_topic()
        else:
            exploration, action, topic = self.explore()
        self.explored_topics.append(topic)
        strategy_analysis, selected_strategies = self.select_strategy(state)
        last_utterance = self.messages[-1]["content"]
        response, final_strategy = self.generate(
            last_utterance, topic, state, selected_strategies
        )
        if final_strategy == "Combined Strategies":
            final_strategy = " + ".join(selected_strategies)
            strategy_description = (
                self.strategy2description[selected_strategies[0]]
                + "\n- "
                + self.strategy2description[selected_strategies[1]]
            )
        else:
            strategy_description = self.strategy2description[final_strategy]
        response = response.replace("\n", " ").strip().lstrip()
        response = self.refine(
            self.conversation[-5:],
            response,
            strategy_description,
            self.topic2description[topic],
        )
        response = response.replace("\n", " ").strip().lstrip()
        self.messages[-1]["content"] = last_utterance
        self.messages.append({"role": "assistant", "content": response})
        self.conversation.append(response)
        return f"[Inferred State: {state} || Strategy Selection: {strategy_analysis} || Strategies: {selected_strategies} || Final Strategy: {final_strategy} || Topic: {topic} || Exploration Action: {action} || Exploration: {exploration}] {response}"
