import backoff
import openai
from openai import OpenAI
import numpy as np
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.Timeout, openai.APIError, openai.APIConnectionError, openai.APIStatusError))
def get_precise_response(messages, model="gpt-4o-mini", temperature=0.2, top_p=0.1):
        message = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
        )
        return message.choices[0].message.content

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.Timeout, openai.APIError, openai.APIConnectionError, openai.APIStatusError))
def get_chatbot_response(messages, model="gpt-4o-mini", temperature=0.7, top_p=0.8, max_tokens=100):
        message = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=100
        )
        return message.choices[0].message.content

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.Timeout, openai.APIError, openai.APIConnectionError, openai.APIStatusError))
def get_json_response(messages, model="gpt-4o-mini", temperature=0.2, top_p=0.1):
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
    def __init__(self, 
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
                 model):
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
        self.last_concern = ''
        self.context = ['Counselor: Hello. How are you?', 'Client: I am good. What about you?']
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
            "Terminate": "You should highlight current state and engagement, express a desire to end the current session, and suggest further discussion be deferred to a later time."
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
            "Chronic Obstructive Pulmonary Disease (COPD)": f"When discussing [topic], The counselor may explore how {self.behavior} can worsen symptoms of COPD, making it harder for you to breathe. They may also explain how {self.goal} can improve your lung function and help manage COPD symptoms.",
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
            "Physical Fitness": f"When discussing [topic], The counselor may talk about how {self.behavior} reduces your physical activity, strength, flexibility, or endurance. The counselor can also discuss how {self.goal} can improve your fitness and overall physical health.",
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
            "Assessment": f"When discussing [topic], The counselor may talk about how {self.behavior} affects your preparation for exams and your overall performance. The counselor can also discuss how {self.goal} can help you improve your focus and achieve better exam results."
        }

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
        personas = '- ' + '\n- '.join(self.personas) + '\n-'.join(self.beliefs)
        system_prompt = system_prompt.replace('[@personas]', personas)
        self.messages = [{'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': f'Counselor: Hello. How are you?'}, 
                         {'role': 'assistant', 'content': 'Client: I am good. What about you?'}]
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
        prompt = prompt.replace('[@goal]', self.goal)
        prompt = prompt.replace('[@context]', '\n- '.join(self.context[-5:]))
        prompt = prompt.replace('[@motivation]', self.motivation)
        response = get_precise_response(messages=[{'role': 'user', 'content': prompt}], model=self.model)
        if "yes" in response.lower():
            self.state = "Motivation"
        return response.split('\n')[0].split(': ')[-1]

    def update_state(self):
        prompt = """## Instruction
You are provided with a dialogue context from a counseling session and a specific target topic. Your task is to evaluate whether the counselor's statements relate to the given topic. Analyze the session and the counselor's responses to determine if they proactively mention the target topic explicitly.
- If the counselor’s statements mention the provided topic explicitly, respond with "Yes."
- If the counselor’s statements do not mention the provided topic explicitly, respond with "No."

Your analysis should focus on whether the counselor has appropriately captured and addressed the target topic.

## Examples

### Example 1
#### Input
Dialogue Context:

- Client: No, I don't think it's a problem at all. It's something I do to help me relax. In fact, I think it's something that is good for me, but
- Counselor: Okay. Can you tell me a little bit more about how, like, it's-it's not an issue for you?
- Client: Yeah. You know, like I said, you know, I'll come home, I'll, you know, smoke a couple rolls after a long day or something. I'm able to just relax, not really think about things. You know what I mean? I'm still working and, you know, paying the bills. We have a house, so I don't- I don't know why it's a problem.
- Counselor: Okay. Um, is it causing any problems in areas like your marriage or anything like that?
- Client: I mean, I guess a little bit, you know, like-- I mean, she's the one that-that has me being here. We've been fighting about it, you know. Uh, uh, I don't know, I-I guess I'm-I'm getting pretty frustrated with her. She just won't get off my back about it. So I finally said, "Fine, I'll go see someone."
- Counselor: Okay. Um, so are you here to maybe work on your marriage? Is that an area that you are wanting to-to work on?

Topic:
- Interpersonal Relationships: When discussing Interpersonal Relationship, the counselor may talk about how smoke weed affects your relationships with family, children, or others. They may focus on how family dynamics, parenting roles, or conflicts are influenced by your actions. The counselor can also discuss how reducing drug use can help strengthen relationships and foster positive connections.

#### Output
Analysis: Based on the conversation and the given topic of "Interpersonal Relationships," the counselor's responses do begin to explore the target topic. The counselor specifically asks if the client's behavior (smoking weed) is causing any problems in their marriage, which aligns with the topic of how smoking weed may affect relationships. The counselor then further explores whether the client is there to work on their marriage, continuing the focus on interpersonal dynamics.
Answer: Yes

### Example 2
#### Input
Dialogue Context:
- Counselor: Okay, well how important would it be for you to-to, uh, do whatever you can to reduce your risks of stroke or heart attack down the line?
- Client: Well, you know, my wife and I wanna retire in about four to five years and we wanna be really active visiting all sorts of places, The Fjords, Norway, things like that so I-I guess having a stroke or a heart attack would really be bad for that so and, you know, we-we-we wanna live our dreams actively.
- Counselor: Yeah, and-and one of the- certainly one of the risks of stroke is paralysis, and if you wanna be active that could be really a problem.
- Client: Yeah, no-no we-we wanna able to be very active.
- Counselor: Okay, well this medicine can really help lower your risk of having a stroke or heart attack so that five years so now when you do retire you can enjoy it without any disability, uh, what are your thoughts about that?

Topic:
- Interpersonal Relationships: When discussing Interpersonal Relationship, the counselor may talk about how smoke weed affects your relationships with family, children, or others. They may focus on how family dynamics, parenting roles, or conflicts are influenced by your actions. The counselor can also discuss how reducing drug use can help strengthen relationships and foster positive connections.

#### Output
The counselor's statements in the conversation are focused entirely on the client’s health, specifically reducing the risks of stroke and heart attack through medication, in order to support an active retirement. There is no mention of interpersonal relationships, family dynamics, or how the client’s behavior may be affecting their relationships with others, which is the core of the given topic. The dialogue centers on the client's personal health goals rather than exploring how their actions impact family or social connections. Therefore, the counselor’s responses do not align with the provided topic of "Interpersonal Relationships."
Answer: No

## Task
### Input
Dialogue Context:
- [@context]

Topic:
- [@topic]

### Output
"""
        prompt = prompt.replace('[@context]', '\n- '.join(self.context[-2:]))
        tmp_prompt = prompt.replace('[@topic]', f'{self.engagemented_topics[0]}: {self.topic2description[self.engagemented_topics[0]]}'.replace('[topic]', self.engagemented_topics[0]))
        response = get_precise_response(messages=[{'role': 'user', 'content': tmp_prompt}], model=self.model)
        if "yes" in response.lower():
            self.engagement = 4
            self.error_topic_count = 0
            motivation_analysis = self.verify_motivation()
            engagement_analysis = response.split('\n')[0].split(': ')[-1]
            return engagement_analysis + ' ' + motivation_analysis
        tmp_prompt = prompt.replace('[@topic]', f'{self.engagemented_topics[1]}: {self.topic2description[self.engagemented_topics[1]]}'.replace('[topic]', self.engagemented_topics[1]))
        response = get_precise_response(messages=[{'role': 'user', 'content': tmp_prompt}], model=self.model)
        if "yes" in response.lower():
            self.engagement = 3
            self.error_topic_count = 0
            return response.split('\n')[0].split(': ')[-1]
        tmp_prompt = prompt.replace('[@topic]', f'{self.engagemented_topics[2]}: {self.topic2description[self.engagemented_topics[2]]}'.replace('[topic]', self.engagemented_topics[2]))
        response = get_precise_response(messages=[{'role': 'user', 'content': tmp_prompt}], model=self.model)
        if "yes" in response.lower():
            self.engagement = 2
            return response.split('\n')[0].split(': ')[-1]
        else:
            self.engagement = 1
            if len(self.context) > 12:
                self.error_topic_count += 1
            return response.split('\n')[0].split(': ')[-1]
    
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
        prompt = prompt.replace('[@context]', '\n'.join(self.context[-3:]).replace('Client:', '**Client**:').replace('Counselor:', '**Counselor**:'))
        context_aware_action_distribution = None
        for _ in range(5):
            response = get_json_response(messages=[{'role': 'user', 'content': prompt}], model=self.model)
            response = response.replace('```', '').replace('json', '')
            try:
                context_aware_action_distribution = eval(response)
            except SyntaxError:
                continue
            if context_aware_action_distribution:
                break
        if not context_aware_action_distribution:
            context_aware_action_distribution = {
                'Deny': 20,
                'Downplay': 20,
                'Blame': 20,
                'Engage': 20,
                'Inform': 20,
            }
        if self.receptivity < 2:
            receptivity_aware_action_distribution = {
                'Deny': 23,
                'Downplay': 28,
                'Blame': 15,
                'Engage': 11,
                'Inform': 22,
            }
        elif self.receptivity < 3:
            receptivity_aware_action_distribution = {
                'Deny': 20,
                'Downplay': 25,
                'Blame': 10,
                'Engage': 15,
                'Inform': 30,
            }
        elif self.receptivity < 4:
            receptivity_aware_action_distribution = {
                'Deny': 19,
                'Downplay': 21,
                'Blame': 11,
                'Engage': 13,
                'Inform': 36,
            }
        elif self.receptivity < 5:
            receptivity_aware_action_distribution = {
                'Deny': 9,
                'Downplay': 20,
                'Blame': 13,
                'Engage': 14,
                'Inform': 44,
            }
        else:
            receptivity_aware_action_distribution = {
                'Deny': 7,
                'Downplay': 13,
                'Blame': 4,
                'Engage': 16,
                'Inform': 60,
            }
        action_distribution = {
            action: context_aware_action_distribution.get(action, 0) + receptivity_aware_action_distribution[action]
            for action in receptivity_aware_action_distribution
        }
        if len(self.personas) == 0:
            action_distribution['Inform'] = 0
        if len(self.beliefs) == 0:
            action_distribution['Blame'] = 0
        # normalize
        action_distribution = {k: v / sum(action_distribution.values()) for k, v in action_distribution.items()}
        sampled_action = np.random.choice(list(action_distribution.keys()), size=1, p=list(action_distribution.values()))[0]
        return sampled_action

    def select_information(self, action):
        messages = []
        if '?' not in self.context[-1]:
            return None
        prompt = """Here is a conversation between Client and Counselor: 
[@conv]

Is there a question in the last utterance of Counselor? Yes or No"""
        prompt = prompt.replace('[@conv]', '\n'.join(self.context[-3:]))
        response = "Yes, there is a question in the last utterance of Counselor."
        messages.append({'role': 'user', 'content': prompt})
        messages.append({'role': 'assistant', 'content': response})
        if action == 'Inform':
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
            prompt = prompt2.replace('[@persona]', persona)
            messages.append({'role': 'user', 'content': prompt})
            response = get_precise_response(messages=messages, model=self.model)
            messages.append({'role': 'assistant', 'content': response})
            if 'yes' in response.lower():
                if action == "Hesitate":
                    self.last_concern = persona
                return persona
        prompt3 = """Craft a persona for the client that responds to queries in a manner that neither conflicts with nor duplicates the attributes of existing personas. Frame the response **in one simple sentence briefly, with 'you' as the subject**."""
        messages.append({'role': 'user', 'content': prompt3})
        response = get_precise_response(messages=messages, model=self.model)
        personas.append(response)
        if action == "Hesitate":
            self.last_concern = persona
        return response
    
    def convert_subject(self, persona):
        prompt = """##Task
Transform the Persona Details into Sentences Using 'You' as the Subject

## Instructions
Please convert the following persona details into sentences where "You" is the subject.

## Persona
[@persona]

## Response Format
Directly reply with the converted sentences."""
        prompt = prompt.replace('[@persona]', persona)
        response = get_precise_response([{'role': 'user', 'content': prompt}], model=self.model)
        if ':' in response:
            response = response.split(':')[-1].strip()
        return response

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
        motivation_analysis = self.update_state()
        motivation_analysis = motivation_analysis.replace('\n', ' ')
        if self.state == "Motivation":
            instruction = f"[{self.motivation} {self.action2prompt['Acknowledge']}]"
            output_instruction = f"[Engage Analysis: {motivation_analysis} {self.motivation} {self.action2prompt['Acknowledge']}]"
            self.state = "Contemplation"
        else:
            engage_instruction = self.get_engage_instruction()
            if self.error_topic_count >= 5:
                action = "Terminate"
            else:
                action = self.select_action()
            if action == "Inform" or action == "Downplay" or action == "Blame" or action == "Hesitate":
                information = self.select_information(action)
                if not information:
                    action = "Hesitate"
                    instruction = f'[{engage_instruction} {self.state2prompt[self.state]} {self.action2prompt[action]} Don\'t show overknowledge and keep your responses concise (no more than 50 words). Don\'t highlight your state explicitly.]'
                    output_instruction = f'[Engage Analysis: {motivation_analysis} Engage Instruction: {engage_instruction} State Instruction: {self.state2prompt[self.state]} Action Instruction: {self.action2prompt[action]}]'
                else:
                    information = self.convert_subject(information)
                    instruction = f'[{engage_instruction} {self.state2prompt[self.state]} {self.action2prompt[action]} You should follow the persona: {information} Don\'t show overknowledge and keep your responses concise (no more than 50 words). Don\'t highlight your state explicitly.]'
                    output_instruction = f'[Engage Analysis: {motivation_analysis} Engage Instruction: {engage_instruction} State Instruction: {self.state2prompt[self.state]} Information {information} Action Instruction: {self.action2prompt[action]}]'
            elif action == "Plan":
                plan = self.select_plan()
                self.acceptable_plans.pop(self.acceptable_plans.index(plan))
                instruction = f'[{engage_instruction} {self.state2prompt[self.state]} {plan} {self.action2prompt[action]} Don\'t show overknowledge, and keep your responses concise (no more than 50 words). Don\'t highlight your state explicitly.]'
                output_instruction = f'[Engage Analysis: {motivation_analysis} Engage Instruction: {engage_instruction} State Instruction: {self.state2prompt[self.state]} {plan} Action Instruction: {self.action2prompt[action]}]'
            else:
                instruction = f'[{engage_instruction} {self.state2prompt[self.state]} {self.action2prompt[action]} Don\'t show overknowledge and keep your responses concise (no more than 50 words). Don\'t highlight your state explicitly.]'
                output_instruction = f'[Engage Analysis: {motivation_analysis} Engage Instruction: {engage_instruction} State Instruction: {self.state2prompt[self.state]} Action Instruction: {self.action2prompt[action]}]'
        instruction = instruction.replace('\n', ' ')
        output_instruction = output_instruction.replace('\n', ' ')
        self.messages.append({"role": "user", "content": f"{self.context[-1]} {instruction}"})
        for _ in range(5):
            response = get_chatbot_response(self.messages, model=self.model)
            if response.startswith('Client: '):
                break
        response = response.replace('\n', ' ')
        response = response.strip().lstrip()
        self.messages.pop(-1)
        self.messages.append({"role": "user", "content": self.context[-1]})
        self.context.append(response)
        self.messages.append({"role": "assistant", "content": f"{response}"})
        return f"{output_instruction} {response}"

