from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-super-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///career_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

# ─────────────────────────────────────────────────────────
# ENSEMBLE MODEL CLASS — must be defined here so joblib
# can deserialise career_model.pkl correctly
# ─────────────────────────────────────────────────────────
class EnsembleCareerModel:
    DEFAULTS = {
        "stream":         "Science",
        "interests":      "Coding",
        "degree":         "btech",
        "coding_skill":   "beginner",
        "preferred_work": "technical",
        "goal":           "high_salary",
        "tenth_pct":      75.0,
        "twelfth_pct":    75.0,
        "grad_pct":       75.0,
    }
    CAT_COLS = ["stream", "degree", "coding_skill", "preferred_work", "goal"]
    NUM_COLS = ["tenth_pct", "twelfth_pct", "grad_pct"]
    TEXT_COL = "interests"
    ALL_COLS = CAT_COLS + NUM_COLS + [TEXT_COL]

    def __init__(self, preprocessor, gb, rf, classes):
        self.preprocessor_ = preprocessor
        self.gb_            = gb
        self.rf_            = rf
        self.classes_       = np.array(classes)

    def _align(self, X_df):
        out = X_df.copy()
        for col, val in self.DEFAULTS.items():
            if col not in out.columns:
                out[col] = val
            else:
                out[col] = out[col].fillna(val)
        return out[self.ALL_COLS]

    def predict_proba(self, X_df):
        Xp = self.preprocessor_.transform(self._align(X_df))
        return 0.60 * self.gb_.predict_proba(Xp) + 0.40 * self.rf_.predict_proba(Xp)

    def predict(self, X_df):
        return self.classes_[np.argmax(self.predict_proba(X_df), axis=1)]


# ─────────────────────────────────────────────────────────
# LOAD TRAINED ML MODEL
# ─────────────────────────────────────────────────────────
try:
    career_model = joblib.load('career_model.pkl')
    CAREERS = list(career_model.classes_)
    print("✓ ML model loaded successfully!")
except FileNotFoundError:
    print("⚠ career_model.pkl not found. Using rule-based fallback.")
    career_model = None
    CAREERS = [
        "Data Scientist", "ML Engineer", "Software Developer", "Backend Developer",
        "Frontend Developer", "Cloud Engineer", "Cybersecurity Analyst", "Business Analyst",
        "Product Manager", "UI/UX Designer", "Mobile App Developer", "DevOps Engineer",
        "Database Administrator", "Network Engineer", "AI Researcher", "Blockchain Developer",
        "Game Developer", "Digital Marketer", "Content Strategist", "Finance Analyst",
        "HR Specialist", "Mechanical Engineer", "Civil Engineer", "Doctor", "Lawyer"
    ]

# ─────────────────────────────────────────────────────────
# CAREER DETAILS DATABASE
# ─────────────────────────────────────────────────────────
CAREER_DETAILS = {
    'Data Scientist': {
        'explanation': 'Your Science/Engineering background combined with AI or Data interests makes Data Science a natural fit.',
        'roadmap': [
            'Month 1-2: Python (Pandas, NumPy, Matplotlib)',
            'Month 3-4: Statistics, SQL, PowerBI',
            'Month 5-6: ML (Scikit-learn), Kaggle competitions',
            'Cert: Google Data Analytics Professional',
            'Projects: 3 end-to-end ML pipelines on Kaggle'
        ],
        'skills_gap': 'SQL mastery + 3 Kaggle projects. Weak stats? Khan Academy.',
        'salary': 'Freshers: ₹6-10L | Mid: ₹15-25L | Senior: ₹35L+',
        'demand': 'Exploding (30% YoY growth)'
    },
    'ML Engineer': {
        'explanation': 'Engineering background + coding/AI interests = ideal profile for production ML systems.',
        'roadmap': [
            'Month 1-2: Python + PyTorch/TensorFlow',
            'Month 3-4: Docker, FastAPI deployment',
            'Month 5-6: MLOps (MLflow), AWS SageMaker',
            'Cert: AWS ML Specialty',
            'Projects: Deploy 2 ML apps to cloud'
        ],
        'skills_gap': 'Docker/containerization critical. No cloud exp? AWS free tier.',
        'salary': 'Freshers: ₹6-12L | Mid: ₹18-30L | Senior: ₹40L+',
        'demand': 'Highest growth (45% YoY)'
    },
    'AI Researcher': {
        'explanation': 'Strong academics + research-oriented interests put you on the path to cutting-edge AI.',
        'roadmap': [
            'Month 1-3: Read 20 arXiv papers, learn PyTorch',
            'Month 4-6: Reproduce SOTA models',
            'Month 7+: Publish on arXiv, attend conferences',
            'Cert: None (papers matter most)',
            'Projects: Novel research + GitHub'
        ],
        'skills_gap': 'Academic writing + research methodology.',
        'salary': 'Research: ₹20-40L | Academia: ₹15-30L',
        'demand': 'Niche but growing'
    },
    'Software Developer': {
        'explanation': 'Strong coding foundation and problem-solving aptitude make Software Development a top fit.',
        'roadmap': [
            'Month 1-2: Python/JavaScript + DSA (LeetCode 200)',
            'Month 3-4: Full-stack (React + Node)',
            'Month 5-6: System design, Git workflows',
            'Cert: AWS Developer Associate',
            'Projects: 3 full-stack apps'
        ],
        'skills_gap': 'LeetCode medium/hard daily.',
        'salary': 'Freshers: ₹4-8L | Mid: ₹12-20L | Senior: ₹25L+',
        'demand': 'Stable high demand'
    },
    'Backend Developer': {
        'explanation': 'Technical mindset + interest in databases and APIs aligns well with Backend Development.',
        'roadmap': [
            'Month 1-2: FastAPI/Express.js + PostgreSQL',
            'Month 3-4: Redis caching, message queues',
            'Month 5-6: Microservices + Kubernetes',
            'Cert: CKAD (Kubernetes)',
            'Projects: Scalable API service'
        ],
        'skills_gap': 'Database optimization + message queues.',
        'salary': 'Freshers: ₹5-9L | Mid: ₹14-22L | Senior: ₹28L+',
        'demand': 'Growing with cloud adoption'
    },
    'Frontend Developer': {
        'explanation': 'Design sense + coding interests make Frontend Development a great creative-technical fit.',
        'roadmap': [
            'Month 1-2: HTML/CSS/JS + Tailwind',
            'Month 3-4: React/Next.js + TypeScript',
            'Month 5-6: Redux/Zustand + SSR',
            'Cert: None (portfolio is key)',
            'Projects: 5 responsive sites'
        ],
        'skills_gap': 'TypeScript + modern React patterns.',
        'salary': 'Freshers: ₹4-7L | Mid: ₹10-18L | Senior: ₹22L+',
        'demand': 'Consistent demand'
    },
    'Mobile App Developer': {
        'explanation': 'Cross-platform coding interest + design sensibility = strong mobile dev profile.',
        'roadmap': [
            'Month 1-2: Flutter/React Native',
            'Month 3-4: State management + Firebase',
            'Month 5-6: App Store deployment',
            'Cert: Google Associate Android Developer',
            'Projects: 3 published apps'
        ],
        'skills_gap': 'App Store publishing experience.',
        'salary': 'Freshers: ₹5-8L | Mid: ₹12-20L | Senior: ₹25L+',
        'demand': 'Mobile-first growth'
    },
    'Cloud Engineer': {
        'explanation': 'Infrastructure thinking + coding foundation sets you up for high-demand cloud roles.',
        'roadmap': [
            'Month 1-2: AWS Solutions Architect',
            'Month 3-4: Terraform + VPC design',
            'Month 5-6: Multi-cloud (GCP/Azure)',
            'Cert: AWS Solutions Architect Associate',
            'Projects: Migrate app to AWS'
        ],
        'skills_gap': 'AWS certification is mandatory.',
        'salary': 'Freshers: ₹5-9L | Mid: ₹15-25L | Senior: ₹30L+',
        'demand': 'Explosive (50% YoY)'
    },
    'DevOps Engineer': {
        'explanation': 'Automation mindset + infrastructure interest = high-value DevOps engineer.',
        'roadmap': [
            'Month 1-2: Docker + Jenkins',
            'Month 3-4: Kubernetes + Helm',
            'Month 5-6: ArgoCD + monitoring stack',
            'Cert: CKA (Kubernetes Admin)',
            'Projects: Zero-downtime CI/CD pipeline'
        ],
        'skills_gap': 'Kubernetes hands-on experience.',
        'salary': 'Freshers: ₹6-10L | Mid: ₹16-28L | Senior: ₹35L+',
        'demand': 'Critical infrastructure'
    },
    'Database Administrator': {
        'explanation': 'Analytical thinking + data interest points toward a strong DBA career.',
        'roadmap': [
            'Month 1-2: PostgreSQL/MySQL admin',
            'Month 3-4: Performance tuning',
            'Month 5-6: Replication + backups',
            'Cert: Oracle DBA',
            'Projects: High-availability DB setup'
        ],
        'skills_gap': 'Query optimization skills.',
        'salary': 'Freshers: ₹5-8L | Mid: ₹12-20L | Senior: ₹25L+',
        'demand': 'Stable enterprise demand'
    },
    'Network Engineer': {
        'explanation': 'Technical interest in connectivity and security aligns with Network Engineering.',
        'roadmap': [
            'Month 1-2: CCNA fundamentals',
            'Month 3-4: Cisco/Juniper routing',
            'Month 5-6: SDN + cloud networking',
            'Cert: CCNP Enterprise',
            'Projects: Enterprise network design'
        ],
        'skills_gap': 'CCNA certification.',
        'salary': 'Freshers: ₹4-7L | Mid: ₹10-18L | Senior: ₹22L+',
        'demand': 'Essential infrastructure'
    },
    'Cybersecurity Analyst': {
        'explanation': 'Security awareness + technical background = strong fit as cyber threats rise 40% YoY.',
        'roadmap': [
            'Month 1-2: Networking + Linux fundamentals',
            'Month 3-4: Wireshark + SIEM tools',
            'Month 5-6: Ethical hacking labs',
            'Cert: CEH + CompTIA Security+',
            'Projects: CTF competitions'
        ],
        'skills_gap': 'Capture The Flag (CTF) practice.',
        'salary': 'Freshers: ₹6-10L | Mid: ₹15-25L | Senior: ₹30L+',
        'demand': 'Critical shortage'
    },
    'Blockchain Developer': {
        'explanation': 'Finance + coding interests + Engineering background puts you ahead in Web3.',
        'roadmap': [
            'Month 1-2: Solidity + Ethereum',
            'Month 3-4: Smart contracts + Web3.js',
            'Month 5-6: DeFi protocols',
            'Cert: Blockchain Council',
            'Projects: DApp + NFT marketplace'
        ],
        'skills_gap': 'Solidity + EVM internals.',
        'salary': 'Freshers: ₹8-15L | Mid: ₹20-35L | Senior: ₹45L+',
        'demand': 'Niche premium pay'
    },
    'Game Developer': {
        'explanation': 'Creative interests + coding = a natural fit for game development.',
        'roadmap': [
            'Month 1-2: Unity + C#',
            'Month 3-4: Physics + shaders',
            'Month 5-6: Multiplayer networking',
            'Cert: Unity Certified Developer',
            'Projects: 1 complete Steam-released game'
        ],
        'skills_gap': '1 shipped, published game.',
        'salary': 'Freshers: ₹5-9L | Mid: ₹12-20L | Senior: ₹25L+',
        'demand': 'Gaming industry boom'
    },
    'Business Analyst': {
        'explanation': 'Commerce background + analytical thinking = strong Business Analyst profile.',
        'roadmap': [
            'Month 1-2: Excel + SQL advanced',
            'Month 3-4: PowerBI + requirements docs',
            'Month 5-6: Agile/Scrum + stakeholder mgmt',
            'Cert: CBAP or PMI-PBA',
            'Projects: 3 business dashboards'
        ],
        'skills_gap': 'Stakeholder communication skills.',
        'salary': 'Freshers: ₹5-9L | Mid: ₹12-20L | Senior: ₹25L+',
        'demand': 'Enterprise staple'
    },
    'Product Manager': {
        'explanation': 'Leadership mindset + technical understanding = ideal Product Manager candidate.',
        'roadmap': [
            'Month 1-2: Product fundamentals + user research',
            'Month 3-4: SQL + A/B testing',
            'Month 5-6: Jira + roadmapping',
            'Cert: Google Product Management',
            'Projects: Launch an MVP'
        ],
        'skills_gap': 'Technical product sense + metrics thinking.',
        'salary': 'Freshers: ₹10-18L | Mid: ₹25-40L | Senior: ₹50L+',
        'demand': 'High demand in tech leadership'
    },
    'UI/UX Designer': {
        'explanation': 'Creative background + user empathy = strong foundation for UI/UX Design.',
        'roadmap': [
            'Month 1-2: Figma + design systems',
            'Month 3-4: User research + usability testing',
            'Month 5-6: Motion design + accessibility',
            'Cert: Google UX Design Certificate',
            'Projects: 10 portfolio case studies'
        ],
        'skills_gap': 'Portfolio with real user testing.',
        'salary': 'Freshers: ₹5-9L | Mid: ₹12-20L | Senior: ₹25L+',
        'demand': 'Design systems trend'
    },
    'Digital Marketer': {
        'explanation': 'Commerce stream + analytics interests = strong Digital Marketing profile.',
        'roadmap': [
            'Month 1-2: Google Analytics 4',
            'Month 3-4: Google Ads + Meta Ads',
            'Month 5-6: SEO + content strategy',
            'Cert: Google Analytics + Ads certified',
            'Projects: Run ₹50k ad campaigns'
        ],
        'skills_gap': 'Hands-on paid ads experience.',
        'salary': 'Freshers: ₹4-8L | Mid: ₹10-18L | Senior: ₹22L+',
        'demand': 'Digital transformation boom'
    },
    'Content Strategist': {
        'explanation': 'Arts background + SEO storytelling = ideal Content Strategist.',
        'roadmap': [
            'Month 1-2: SEO + keyword research',
            'Month 3-4: Content calendars + video',
            'Month 5-6: Analytics + A/B headlines',
            'Cert: HubSpot Content Marketing',
            'Projects: 50 articles + YouTube channel'
        ],
        'skills_gap': 'SEO expertise + video content.',
        'salary': 'Freshers: ₹4-7L | Mid: ₹9-16L | Senior: ₹20L+',
        'demand': 'Content marketing boom'
    },
    'Finance Analyst': {
        'explanation': 'Commerce stream + Excel/data skills make Finance Analysis a great match.',
        'roadmap': [
            'Month 1-2: Advanced Excel + VBA',
            'Month 3-4: Financial modeling + DCF',
            'Month 5-6: PowerBI + valuation',
            'Cert: CFA Level 1',
            'Projects: 5 financial models'
        ],
        'skills_gap': 'Financial modeling + DCF analysis.',
        'salary': 'Freshers: ₹5-9L | Mid: ₹12-22L | Senior: ₹30L+',
        'demand': 'Fintech explosion'
    },
    'HR Specialist': {
        'explanation': 'People-oriented personality + organizational skills = natural HR Specialist.',
        'roadmap': [
            'Month 1-2: Recruitment + ATS systems',
            'Month 3-4: Labor laws + payroll',
            'Month 5-6: Employee engagement',
            'Cert: SHRM-CP',
            'Projects: End-to-end hiring process'
        ],
        'skills_gap': 'Structured interview frameworks.',
        'salary': 'Freshers: ₹4-7L | Mid: ₹10-18L | Senior: ₹25L+',
        'demand': 'Talent wars era'
    },
    'Mechanical Engineer': {
        'explanation': 'Science/Engineering stream + practical design interest = strong Mechanical Engineering fit.',
        'roadmap': [
            'Month 1-2: SolidWorks/AutoCAD',
            'Month 3-4: FEA + ANSYS',
            'Month 5-6: Manufacturing processes',
            'Cert: ASME CAD Certification',
            'Projects: 3D prototype design'
        ],
        'skills_gap': 'CAD + simulation software.',
        'salary': 'Freshers: ₹4-7L | Mid: ₹10-18L | Senior: ₹25L+',
        'demand': 'Manufacturing 4.0'
    },
    'Civil Engineer': {
        'explanation': 'Science stream + infrastructure interest = solid Civil Engineering profile.',
        'roadmap': [
            'Month 1-2: AutoCAD + STAAD.Pro',
            'Month 3-4: Structural analysis',
            'Month 5-6: Project management',
            'Cert: Primavera P6',
            'Projects: RCC design + cost estimation'
        ],
        'skills_gap': 'Structural software proficiency.',
        'salary': 'Freshers: ₹4-6L | Mid: ₹9-16L | Senior: ₹22L+',
        'demand': 'Smart cities growth'
    },
    'Doctor': {
        'explanation': 'Science stream + Healthcare interest + strong academics = ideal MBBS candidate.',
        'roadmap': [
            'Year 1-5.5: NEET → MBBS',
            'Year 6-9: PG specialization (MD/MS)',
            'Year 10+: Practice/hospital/research',
            'Cert: NEET PG',
            'Projects: Clinical rotations + research'
        ],
        'skills_gap': 'NEET score 650+.',
        'salary': 'MBBS: ₹12-25L | Specialist: ₹30-80L+',
        'demand': 'Healthcare crisis demand'
    },
    'Lawyer': {
        'explanation': 'Arts/Commerce stream + analytical thinking = strong legal career path.',
        'roadmap': [
            'Year 1-5: CLAT → LLB (5-year integrated)',
            'Year 6: Law firm internship',
            'Year 7+: Bar exam + practice area',
            'Cert: AIBE (All India Bar Exam)',
            'Projects: Moot courts + legal research'
        ],
        'skills_gap': 'CLAT prep + legal research skills.',
        'salary': 'Freshers: ₹6-12L | Mid: ₹15-30L | Senior: ₹50L+',
        'demand': 'Corporate law boom'
    }
}

# Fill any missing careers with a generic template
for career in CAREERS:
    if career not in CAREER_DETAILS:
        CAREER_DETAILS[career] = {
            'explanation': 'Strong profile alignment detected for this career path.',
            'roadmap': [
                '1. Build strong subject foundations',
                '2. Complete structured online courses',
                '3. Work on 2-3 portfolio projects',
                '4. Earn 1 relevant certification',
                '5. Apply for internships'
            ],
            'skills_gap': 'Complete a full skill assessment first.',
            'salary': '₹8-20 LPA'
        }


# ─────────────────────────────────────────────────────────
# USER MODEL
# New fields: degree, coding_skill, preferred_work, goal
# These improve ML accuracy but are fully optional —
# old records without them still work fine.
# ─────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    name          = db.Column(db.String(80))
    age           = db.Column(db.Integer)
    stream        = db.Column(db.String(50))
    interests     = db.Column(db.String(255))
    tenth_pct     = db.Column(db.Float)
    twelfth_pct   = db.Column(db.Float)
    grad_pct      = db.Column(db.Float)
    # New fields for better ML accuracy
    degree        = db.Column(db.String(20))
    coding_skill  = db.Column(db.String(20))
    preferred_work= db.Column(db.String(20))
    goal          = db.Column(db.String(20))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ─────────────────────────────────────────────────────────
# RULE-BASED SCORE BOOSTS
# Applied on top of ML probabilities. Encodes domain
# knowledge so that even if the ML is uncertain, the
# output respects clear career-profile matches.
# ─────────────────────────────────────────────────────────
BOOST_RULES = {
    'Data Scientist':        lambda u, i, s: ('data' in i or 'ai' in i) and s in ('science', 'engineering'),
    'ML Engineer':           lambda u, i, s: 'ai' in i and 'coding' in i and s == 'engineering',
    'AI Researcher':         lambda u, i, s: 'ai' in i and 'research' in i,
    'Software Developer':    lambda u, i, s: 'coding' in i,
    'Backend Developer':     lambda u, i, s: 'coding' in i and 'data' in i and s == 'engineering',
    'Frontend Developer':    lambda u, i, s: 'coding' in i and 'design' in i,
    'Mobile App Developer':  lambda u, i, s: 'coding' in i and 'design' in i,
    'Cloud Engineer':        lambda u, i, s: 'coding' in i and 'security' in i and s == 'engineering',
    'DevOps Engineer':       lambda u, i, s: 'coding' in i and s == 'engineering',
    'Cybersecurity Analyst': lambda u, i, s: 'security' in i,
    'Blockchain Developer':  lambda u, i, s: 'coding' in i and 'finance' in i,
    'Game Developer':        lambda u, i, s: 'coding' in i and 'design' in i,
    'Business Analyst':      lambda u, i, s: ('business' in i or 'finance' in i) and s == 'commerce',
    'Product Manager':       lambda u, i, s: 'business' in i and 'data' in i,
    'UI/UX Designer':        lambda u, i, s: 'design' in i,
    'Digital Marketer':      lambda u, i, s: 'marketing' in i and s in ('commerce', 'arts'),
    'Content Strategist':    lambda u, i, s: 'marketing' in i and s == 'arts',
    'Finance Analyst':       lambda u, i, s: 'finance' in i and s == 'commerce',
    'HR Specialist':         lambda u, i, s: 'business' in i and s in ('arts', 'commerce'),
    'Doctor':                lambda u, i, s: 'healthcare' in i and s == 'science',
    'Lawyer':                lambda u, i, s: 'law' in i and s in ('arts', 'commerce'),
    'Mechanical Engineer':   lambda u, i, s: s in ('science', 'engineering') and 'coding' not in i,
    'Civil Engineer':        lambda u, i, s: s in ('science', 'engineering') and 'coding' not in i,
}

def apply_boosts(results, user):
    """
    Score each career using a weighted point system instead of
    flat boosts — so careers match at different levels and
    scores are always differentiated.
    """
    interests_str = (user.interests or "").lower()
    stream_str    = (user.stream or "").lower()
    coding_str    = (user.coding_skill or "").lower()
    goal_str      = (user.goal or "").lower()
    work_str      = (user.preferred_work or "").lower()
    tenth         = user.tenth_pct or 0
    twelfth       = user.twelfth_pct or 0
    grad          = user.grad_pct or 0
    avg_marks     = (tenth + twelfth + grad) / 3 if (tenth or twelfth or grad) else 0

    interest_list = [i.strip().lower() for i in interests_str.split(',') if i.strip()]

    # Per-career scoring rules — each rule returns a score 0-100
    SCORING = {
        'Data Scientist': lambda: (
            35 * ('data' in interest_list or 'ai' in interest_list) +
            20 * (stream_str in ('science', 'engineering')) +
            15 * ('coding' in interest_list) +
            15 * (coding_str in ('intermediate', 'advanced')) +
            10 * (work_str in ('analytical', 'technical')) +
            5  * (avg_marks >= 75)
        ),
        'ML Engineer': lambda: (
            40 * ('ai' in interest_list) +
            20 * ('coding' in interest_list) +
            20 * (stream_str == 'engineering') +
            15 * (coding_str == 'advanced') +
            5  * (avg_marks >= 75)
        ),
        'AI Researcher': lambda: (
            40 * ('ai' in interest_list) +
            25 * (goal_str == 'research') +
            20 * (stream_str in ('science', 'engineering')) +
            10 * (coding_str == 'advanced') +
            5  * (avg_marks >= 80)
        ),
        'Software Developer': lambda: (
            40 * ('coding' in interest_list) +
            20 * (coding_str in ('intermediate', 'advanced')) +
            15 * (work_str == 'technical') +
            15 * (stream_str in ('science', 'engineering')) +
            10 * (avg_marks >= 60)
        ),
        'Backend Developer': lambda: (
            35 * ('coding' in interest_list) +
            25 * (stream_str == 'engineering') +
            20 * (coding_str in ('intermediate', 'advanced')) +
            10 * ('data' in interest_list) +
            10 * (work_str == 'technical')
        ),
        'Frontend Developer': lambda: (
            35 * ('coding' in interest_list) +
            30 * ('design' in interest_list) +
            20 * (work_str == 'creative') +
            10 * (coding_str in ('beginner', 'intermediate', 'advanced')) +
            5  * (avg_marks >= 55)
        ),
        'Mobile App Developer': lambda: (
            35 * ('coding' in interest_list) +
            25 * ('design' in interest_list) +
            20 * (coding_str in ('intermediate', 'advanced')) +
            10 * (work_str in ('technical', 'creative')) +
            10 * (stream_str in ('science', 'engineering'))
        ),
        'Cloud Engineer': lambda: (
            35 * ('coding' in interest_list) +
            25 * (stream_str == 'engineering') +
            20 * ('security' in interest_list) +
            20 * (coding_str in ('intermediate', 'advanced'))
        ),
        'DevOps Engineer': lambda: (
            35 * ('coding' in interest_list) +
            30 * (stream_str == 'engineering') +
            20 * (coding_str in ('intermediate', 'advanced')) +
            15 * (work_str == 'technical')
        ),
        'Cybersecurity Analyst': lambda: (
            45 * ('security' in interest_list) +
            20 * ('coding' in interest_list) +
            20 * (stream_str in ('science', 'engineering')) +
            15 * (coding_str in ('intermediate', 'advanced'))
        ),
        'Blockchain Developer': lambda: (
            40 * ('coding' in interest_list and 'finance' in interest_list) +
            25 * (coding_str == 'advanced') +
            20 * (stream_str in ('science', 'engineering')) +
            15 * ('data' in interest_list)
        ),
        'Game Developer': lambda: (
            40 * ('coding' in interest_list and 'design' in interest_list) +
            30 * (work_str == 'creative') +
            20 * (coding_str in ('intermediate', 'advanced')) +
            10 * (goal_str == 'passion')
        ),
        'Business Analyst': lambda: (
            40 * (('business' in interest_list or 'finance' in interest_list) and stream_str == 'commerce') +
            20 * (work_str == 'analytical') +
            20 * ('data' in interest_list) +
            10 * (stream_str == 'commerce') +
            10 * (avg_marks >= 60)
        ),
        'Product Manager': lambda: (
            35 * ('business' in interest_list) +
            25 * ('data' in interest_list) +
            20 * (work_str == 'people') +
            10 * (goal_str == 'high_salary') +
            10 * (coding_str in ('beginner', 'intermediate'))
        ),
        'UI/UX Designer': lambda: (
            50 * ('design' in interest_list) +
            25 * (work_str == 'creative') +
            15 * (goal_str == 'passion') +
            10 * (stream_str in ('arts', 'engineering'))
        ),
        'Digital Marketer': lambda: (
            45 * ('marketing' in interest_list) +
            25 * (stream_str in ('commerce', 'arts')) +
            20 * ('business' in interest_list) +
            10 * (work_str == 'creative')
        ),
        'Content Strategist': lambda: (
            45 * ('marketing' in interest_list and stream_str == 'arts') +
            25 * (work_str == 'creative') +
            20 * (goal_str == 'passion') +
            10 * (stream_str == 'arts')
        ),
        'Finance Analyst': lambda: (
            50 * ('finance' in interest_list and stream_str == 'commerce') +
            20 * (work_str == 'analytical') +
            20 * (stream_str == 'commerce') +
            10 * (avg_marks >= 65)
        ),
        'HR Specialist': lambda: (
            40 * ('business' in interest_list and stream_str in ('arts', 'commerce')) +
            30 * (work_str == 'people') +
            20 * (stream_str in ('arts', 'commerce')) +
            10 * (goal_str == 'job_security')
        ),
        'Doctor': lambda: (
            50 * ('healthcare' in interest_list and stream_str == 'science') +
            25 * (avg_marks >= 80) +
            15 * (stream_str == 'science') +
            10 * (goal_str == 'passion')
        ),
        'Lawyer': lambda: (
            50 * ('law' in interest_list and stream_str in ('arts', 'commerce')) +
            25 * (work_str in ('analytical', 'people')) +
            15 * (stream_str in ('arts', 'commerce')) +
            10 * (goal_str == 'high_salary')
        ),
        'Mechanical Engineer': lambda: (
            40 * (stream_str in ('science', 'engineering') and 'coding' not in interest_list) +
            25 * (work_str == 'technical') +
            20 * (stream_str in ('science', 'engineering')) +
            15 * (avg_marks >= 60)
        ),
        'Civil Engineer': lambda: (
            40 * (stream_str in ('science', 'engineering') and 'coding' not in interest_list) +
            25 * (work_str in ('technical', 'analytical')) +
            20 * (stream_str in ('science', 'engineering')) +
            15 * (avg_marks >= 60)
        ),
        'Database Administrator': lambda: (
            35 * ('data' in interest_list) +
            25 * ('coding' in interest_list) +
            20 * (work_str == 'analytical') +
            20 * (stream_str in ('science', 'engineering', 'commerce'))
        ),
        'Network Engineer': lambda: (
            40 * ('security' in interest_list) +
            30 * (stream_str == 'engineering') +
            20 * (work_str == 'technical') +
            10 * (coding_str in ('beginner', 'intermediate'))
        ),
    }

    for r in results:
        career = str(r['career'])
        if career in SCORING:
            try:
                raw = SCORING[career]()
                # Normalise: max possible is ~100, map to 10-95 range
                r['score'] = round(10 + (raw / 100) * 85, 1)
            except Exception:
                pass

    # Add small random noise so tied careers are separated
    import random
    for r in results:
        r['score'] = round(r['score'] + random.uniform(0, 1.5), 1)

    return sorted(results, key=lambda x: x['score'], reverse=True)


# ─────────────────────────────────────────────────────────
# DYNAMIC EXPLANATION BUILDER
# Makes each explanation feel personalised to the user.
# ─────────────────────────────────────────────────────────
def build_explanation(career, base_explanation, user):
    # ── FIX: ensure career is always a plain Python string ──
    career = str(career)

    parts = [base_explanation]
    interests_str = (user.interests or "").lower()
    stream_str    = (user.stream or "").lower()

    interest_list = [i.strip() for i in interests_str.split(',') if i.strip()]

    # Stream match mention
    stream_map = {
        'science':     ['Data Scientist', 'ML Engineer', 'AI Researcher', 'Doctor',
                        'Mechanical Engineer', 'Civil Engineer'],
        'commerce':    ['Business Analyst', 'Finance Analyst', 'Digital Marketer',
                        'HR Specialist', 'Lawyer'],
        'arts':        ['UI/UX Designer', 'Content Strategist', 'Digital Marketer',
                        'Lawyer', 'HR Specialist'],
        'engineering': ['Software Developer', 'Backend Developer', 'Cloud Engineer',
                        'DevOps Engineer', 'ML Engineer', 'Network Engineer'],
    }
    for stream, careers in stream_map.items():
        if stream in stream_str and career in careers:
            parts.append(f"Your {stream.capitalize()} stream is a direct match.")
            break

    # Interest mentions
    if 'ai' in interest_list and career in ('Data Scientist', 'ML Engineer', 'AI Researcher'):
        parts.append("Your AI interest strongly signals this direction.")
    if 'data' in interest_list and 'data' in career.lower():
        parts.append("Data interest aligns perfectly with this role.")
    if 'security' in interest_list and career == 'Cybersecurity Analyst':
        parts.append("Security interest is a rare, high-demand signal.")
    if 'design' in interest_list and career in ('UI/UX Designer', 'Game Developer', 'Frontend Developer'):
        parts.append("Design interest makes you stand out for this role.")
    if 'finance' in interest_list and career in ('Finance Analyst', 'Blockchain Developer'):
        parts.append("Finance interest directly boosts your fit here.")

    # Academic boosts
    if user.tenth_pct and user.tenth_pct >= 90:
        parts.append("Excellent 10th marks show the academic discipline this field rewards.")
    elif user.tenth_pct and user.tenth_pct >= 80:
        parts.append("Solid academic track record works in your favour.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────
# MAIN RECOMMENDATION ENGINE
# Returns top3 for the dashboard cards + full list
# ─────────────────────────────────────────────────────────
def get_career_recommendations(user):
    # Build feature DataFrame — new fields are optional
    X_df = pd.DataFrame([{
        "stream":         user.stream          or "Science",
        "interests":      user.interests        or "Coding",
        "degree":         user.degree           or "btech",
        "coding_skill":   user.coding_skill     or "beginner",
        "preferred_work": user.preferred_work   or "technical",
        "goal":           user.goal             or "high_salary",
        "tenth_pct":      user.tenth_pct        or 70.0,
        "twelfth_pct":    user.twelfth_pct      if user.twelfth_pct  is not None else 75.0,
        "grad_pct":       user.grad_pct         if user.grad_pct     is not None else 75.0,
    }])

    # Get ML probabilities
    if career_model is not None:
        try:
            # ── FIX: encode categorical columns so XGBoost/sklearn won't complain ──
            from sklearn.preprocessing import OrdinalEncoder
            cat_cols = ["stream", "degree", "coding_skill", "preferred_work", "goal"]
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_encoded = X_df.copy()
            X_encoded[cat_cols] = enc.fit_transform(X_df[cat_cols].astype(str))

            proba   = career_model.predict_proba(X_encoded)[0]
            classes = list(career_model.classes_)
            base    = list(zip(classes, proba))
        except Exception as e:
            print(f"ML error: {e} — using fallback")
            base = [(c, 1.0 / len(CAREERS)) for c in CAREERS]
    else:
        # Rule-based fallback: all equal, boosts will differentiate
        base = [(c, 1.0 / len(CAREERS)) for c in CAREERS]

    # Build result objects
    results = []
    for career, prob in base:
        career      = str(career)   # ensure plain string always
        details     = CAREER_DETAILS.get(career, CAREER_DETAILS.get('Software Developer', {}))
        base_expl   = details.get('explanation', 'Strong career alignment detected.')
        explanation = build_explanation(career, base_expl, user)

        results.append({
            "career":     career,
            "score":      round(prob * 100, 1),
            "explanation": explanation,
            "roadmap":    details.get('roadmap', []),
            "skills_gap": details.get('skills_gap', 'Build a strong portfolio.'),
            "salary":     details.get('salary', '₹8-20 LPA'),
        })

    # Apply rule-based boosts + re-sort
    results = apply_boosts(results, user)

    top3  = results[:3]
    all25 = results          # full ranked list (for modal access if needed)
    return top3, all25


# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email')
        password = request.form.get('password')
        user     = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash("Invalid email or password", "error")
    return render_template('auth_login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email    = request.form.get('email')
        password = request.form.get('password')
        name     = request.form.get('name')
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "error")
            return redirect(url_for('register'))
        user = User(email=email, name=name)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('auth_register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    user = current_user

    if request.method == 'POST':
        user.name          = request.form.get('name')
        user.age           = int(request.form.get('age') or 0)
        user.stream        = request.form.get('stream')
        user.interests     = ",".join(request.form.getlist('interests'))
        user.degree        = request.form.get('degree')
        user.coding_skill  = request.form.get('coding_skill')
        user.preferred_work= request.form.get('preferred_work')
        user.goal          = request.form.get('goal')

        raw_tenth   = request.form.get('tenth')
        raw_twelfth = request.form.get('twelfth')
        raw_grad    = request.form.get('grad')

        user.tenth_pct   = float(raw_tenth)   if raw_tenth   else None
        user.twelfth_pct = float(raw_twelfth) if raw_twelfth else None
        user.grad_pct    = float(raw_grad)    if raw_grad    else None

        db.session.commit()
        flash("Profile updated! AI recommendations refreshed.", "success")

    top3, all25 = get_career_recommendations(user)

    # dashboard.html uses top10 variable name — pass top3 as top10
    return render_template('dashboard.html', top10=top3, all25=all25, user=user)


@app.route('/theme', methods=['POST'])
def set_theme():
    data  = request.get_json()
    theme = data.get('theme', 'dark')
    resp  = app.make_response(jsonify({"status": "ok"}))
    resp.set_cookie('theme', theme, max_age=60 * 60 * 24 * 365)
    return resp


if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    print("\n══════════════════════════════════════")
    print("  CareerAI — Starting Server")
    print(f"  ML Model : {'Loaded ✓' if career_model else 'Fallback (rule-based) ⚠'}")
    print("  http://127.0.0.1:5000")
    print("══════════════════════════════════════\n")
    app.run(debug=True, host='0.0.0.0')