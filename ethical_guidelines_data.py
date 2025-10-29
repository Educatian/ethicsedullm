"""
Comprehensive ethical guidelines dataset covering major international frameworks
"""

ETHICAL_GUIDELINES_DATASET = [
    # EU AI Act
    {
        "question": "What are the key risk categories in the EU AI Act?",
        "answer": "The EU AI Act categorizes AI systems into four risk levels: 1) Unacceptable risk (banned applications like social scoring), 2) High-risk systems (e.g., critical infrastructure, law enforcement, employment decisions) requiring strict compliance, 3) Limited risk systems (e.g., chatbots) with transparency obligations, and 4) Minimal risk systems with no specific requirements. High-risk systems must meet requirements for data quality, documentation, transparency, human oversight, and robustness.",
        "source": "EU AI Act",
        "category": "regulation"
    },
    {
        "question": "What obligations do high-risk AI systems have under the EU AI Act?",
        "answer": "High-risk AI systems under the EU AI Act must: establish risk management systems, ensure high-quality training data, maintain technical documentation, enable logging and traceability, provide clear user information, ensure human oversight capabilities, meet cybersecurity and robustness standards, and undergo conformity assessments before market deployment. Providers must also register systems in an EU database and implement post-market monitoring.",
        "source": "EU AI Act",
        "category": "regulation"
    },

    # OECD AI Principles
    {
        "question": "What are the OECD AI Principles?",
        "answer": "The OECD AI Principles, adopted in 2019, include five values-based principles: 1) Inclusive growth, sustainable development, and well-being, 2) Human-centered values and fairness, 3) Transparency and explainability, 4) Robustness, security and safety, and 5) Accountability. These principles emphasize that AI should benefit people and the planet, respect rule of law and human rights, be understandable, function reliably, and have clear responsibility mechanisms.",
        "source": "OECD AI Principles",
        "category": "framework"
    },
    {
        "question": "How does the OECD define AI system accountability?",
        "answer": "The OECD defines accountability in AI as ensuring that organizations and individuals developing, deploying, or operating AI systems are responsible for their proper functioning in line with the principles. This includes establishing mechanisms for redress, having clear documentation, implementing governance frameworks, enabling traceability of decisions, and ensuring there are identifiable parties responsible for AI outcomes and impacts.",
        "source": "OECD AI Principles",
        "category": "framework"
    },

    # Partnership on AI
    {
        "question": "What are the core tenets of Partnership on AI?",
        "answer": "Partnership on AI focuses on eight thematic pillars: 1) Safety-critical AI, 2) Fair, transparent, and accountable AI, 3) Collaboration between people and AI systems, 4) Social and economic implications, 5) AI, labor, and the economy, 6) AI and social good, 7) Ethical research norms, and 8) Accessible AI technologies. The partnership emphasizes multi-stakeholder collaboration, responsible AI development, and ensuring AI benefits society broadly.",
        "source": "Partnership on AI",
        "category": "framework"
    },
    {
        "question": "How does Partnership on AI approach fairness in AI systems?",
        "answer": "Partnership on AI advocates for fairness through multiple approaches: conducting systematic bias audits, using diverse and representative datasets, involving affected communities in design processes, implementing multiple fairness metrics appropriate to context, maintaining transparency about limitations, establishing feedback mechanisms, and continuously monitoring for discriminatory outcomes. They emphasize that fairness is context-dependent and requires ongoing attention throughout the AI lifecycle.",
        "source": "Partnership on AI",
        "category": "framework"
    },

    # UNESCO Ethical Guidelines
    {
        "question": "What are UNESCO's Recommendation on the Ethics of AI key values?",
        "answer": "UNESCO's ethical framework for AI is built on ten core values: 1) Human rights and human dignity, 2) Living in peaceful, just and interconnected societies, 3) Ensuring diversity and inclusiveness, 4) Environment and ecosystem flourishing, 5) Transparency and explainability, 6) Responsibility and accountability, 7) Awareness and literacy, 8) Multi-stakeholder governance, 9) Adaptiveness, and 10) Sustainable development. These values emphasize AI's role in advancing human welfare while protecting fundamental rights.",
        "source": "UNESCO Ethical Guidelines",
        "category": "framework"
    },
    {
        "question": "How does UNESCO address environmental concerns in AI ethics?",
        "answer": "UNESCO's guidelines explicitly recognize AI's environmental impact, calling for: minimizing AI systems' carbon footprint and energy consumption, considering the full lifecycle environmental costs including hardware production and disposal, promoting green AI research and development, using AI to address environmental challenges and climate change, ensuring AI development doesn't compromise ecosystem health, and prioritizing sustainable computing practices. This makes UNESCO unique in prominently featuring environmental ethics in AI guidelines.",
        "source": "UNESCO Ethical Guidelines",
        "category": "framework"
    },

    # Cross-cutting ethical concepts
    {
        "question": "What is algorithmic bias and how do major frameworks address it?",
        "answer": "Algorithmic bias occurs when AI systems produce systematically prejudiced results due to flawed assumptions in the ML process. Major frameworks address this through: EU AI Act requires high-quality, representative training data and bias monitoring; OECD emphasizes fairness and inclusive design; Partnership on AI advocates systematic bias audits and diverse teams; UNESCO calls for diversity, inclusiveness, and non-discrimination. All frameworks agree on using diverse data, testing for bias, involving affected communities, and maintaining transparency about limitations.",
        "source": "Multiple frameworks",
        "category": "concept"
    },
    {
        "question": "How do international frameworks define AI transparency?",
        "answer": "Transparency in AI is consistently emphasized across frameworks but with nuanced approaches: EU AI Act mandates technical documentation and user disclosure for high-risk systems; OECD calls for explainability appropriate to context and stakeholder needs; Partnership on AI emphasizes clear communication about capabilities and limitations; UNESCO highlights transparency as essential for accountability. Common elements include: explainability of decisions, disclosure of AI use, documentation of development processes, and accessible information for affected parties.",
        "source": "Multiple frameworks",
        "category": "concept"
    },
    {
        "question": "What is the right to human oversight in AI systems?",
        "answer": "Human oversight means maintaining meaningful human control over AI systems, especially for consequential decisions. This principle appears across frameworks: EU AI Act requires human oversight mechanisms for high-risk systems with ability to override decisions; OECD emphasizes human agency and oversight; Partnership on AI advocates for 'human-in-the-loop' approaches; UNESCO stresses human autonomy and decision-making. Key aspects include: humans can intervene, understand system outputs, recognize system limitations, and maintain ultimate authority over critical decisions.",
        "source": "Multiple frameworks",
        "category": "concept"
    },

    # Fairness definitions and metrics
    {
        "question": "What are the different mathematical definitions of fairness in machine learning?",
        "answer": "Major fairness definitions include: 1) Demographic parity (equal positive prediction rates across groups), 2) Equalized odds (equal true positive and false positive rates), 3) Equal opportunity (equal true positive rates), 4) Predictive parity (equal precision across groups), 5) Individual fairness (similar individuals receive similar predictions), and 6) Calibration (predicted probabilities match actual outcomes). These definitions can conflict—satisfying one may violate others. The choice depends on the application context, stakeholder values, and legal requirements. Frameworks recommend considering multiple metrics and trade-offs.",
        "source": "Technical ML ethics literature",
        "category": "concept"
    },
    {
        "question": "What is the difference between group fairness and individual fairness?",
        "answer": "Group fairness ensures statistical parity or error rates are equal across demographic groups (e.g., equal approval rates for loans across races). Individual fairness requires that similar individuals be treated similarly, regardless of group membership. Group fairness is easier to measure but can mask individual discrimination; individual fairness is conceptually appealing but requires defining 'similarity,' which can be subjective. Modern approaches often combine both: ensuring group-level equity while minimizing individual-level disparities. The appropriate balance depends on legal context, stakeholder input, and application domain.",
        "source": "Technical ML ethics literature",
        "category": "concept"
    },

    # Privacy and data protection
    {
        "question": "How do AI ethics frameworks address privacy and data protection?",
        "answer": "Privacy is central across frameworks: EU AI Act requires GDPR compliance and data minimization for high-risk systems; OECD calls for protecting privacy and data security throughout AI lifecycles; Partnership on AI emphasizes privacy-preserving techniques and user control; UNESCO highlights privacy as a human right requiring protection. Common recommendations include: data minimization, purpose limitation, secure storage, privacy-by-design, user consent, differential privacy techniques, federated learning where appropriate, and clear data governance policies.",
        "source": "Multiple frameworks",
        "category": "concept"
    },
    {
        "question": "What is differential privacy and why is it important for AI ethics?",
        "answer": "Differential privacy is a mathematical framework that adds carefully calibrated noise to data or query results, ensuring that individual records cannot be identified even if attackers have auxiliary information. It's important for AI ethics because it allows learning from sensitive datasets while providing formal privacy guarantees. Applications include census data, medical research, and recommendation systems. However, it involves accuracy-privacy trade-offs: stronger privacy guarantees reduce model utility. Ethical implementation requires transparent communication about privacy levels and limitations, user understanding of protections, and appropriate calibration for the sensitivity of data.",
        "source": "Technical privacy literature",
        "category": "concept"
    },

    # Explainability and interpretability
    {
        "question": "What is the difference between explainability and interpretability in AI?",
        "answer": "Interpretability refers to the degree to which humans can understand how a model works internally (e.g., decision trees are inherently interpretable). Explainability refers to techniques that provide understandable reasons for specific predictions (e.g., LIME, SHAP for black-box models). Interpretable models are transparent by design; explainable models use post-hoc methods to clarify opaque models. Frameworks vary: EU AI Act requires explanations for high-risk systems; OECD emphasizes context-appropriate transparency; UNESCO calls for understanding AI decision-making. Trade-offs exist between model performance and interpretability—highly accurate models are often less interpretable.",
        "source": "Technical XAI literature",
        "category": "concept"
    },
    {
        "question": "What are the main approaches to explainable AI (XAI)?",
        "answer": "Main XAI approaches include: 1) Inherently interpretable models (linear regression, decision trees, rule-based systems), 2) Model-agnostic methods (LIME, SHAP, counterfactual explanations), 3) Model-specific techniques (attention visualization for neural networks, feature importance for random forests), 4) Example-based explanations (showing similar training examples), and 5) Natural language explanations. Choice depends on audience (technical experts vs. end users), stakes (high-stakes decisions need stronger explanations), regulatory requirements, and trade-offs between accuracy and interpretability. Ethical XAI requires explanations be accurate, complete, and understandable to intended audiences.",
        "source": "Technical XAI literature",
        "category": "concept"
    },

    # Accountability and governance
    {
        "question": "What does AI accountability mean in practice?",
        "answer": "AI accountability in practice involves: 1) Clear assignment of responsibilities throughout the AI lifecycle, 2) Documentation of design decisions, data sources, and model limitations, 3) Audit trails for decisions and changes, 4) Impact assessments before deployment, 5) Monitoring systems for performance and fairness post-deployment, 6) Mechanisms for redress when systems cause harm, 7) Regular third-party audits where appropriate, and 8) Clear communication with stakeholders. Frameworks emphasize that accountability requires both technical measures (logging, testing) and organizational structures (governance, oversight, complaint mechanisms).",
        "source": "Multiple frameworks",
        "category": "concept"
    },
    {
        "question": "What is an Algorithmic Impact Assessment (AIA)?",
        "answer": "An Algorithmic Impact Assessment is a systematic evaluation conducted before deploying AI systems to identify potential harms and benefits. Inspired by privacy and environmental impact assessments, AIAs examine: intended use and stakeholders affected, potential benefits and risks, fairness and bias concerns, privacy implications, transparency and explainability, accountability mechanisms, and mitigation strategies. The EU AI Act mandates similar assessments for high-risk systems. Effective AIAs involve diverse stakeholders, are documented transparently, inform design decisions, and are revisited regularly. They're proactive tools for responsible AI development.",
        "source": "AI governance literature",
        "category": "concept"
    },

    # Safety and robustness
    {
        "question": "What are the key safety concerns for AI systems?",
        "answer": "Key AI safety concerns include: 1) Robustness—performance under distribution shift, adversarial inputs, or edge cases, 2) Security—protection against manipulation, poisoning, or unauthorized access, 3) Reliability—consistent, predictable behavior, 4) Safe failure modes—graceful degradation rather than catastrophic failure, 5) Value alignment—systems pursue intended objectives, 6) Unintended consequences—avoiding negative side effects. Frameworks emphasize: rigorous testing across diverse scenarios, adversarial robustness evaluation, security audits, monitoring in deployment, incident response plans, and human oversight for critical applications.",
        "source": "Multiple frameworks",
        "category": "concept"
    },
    {
        "question": "What is adversarial robustness in AI and why does it matter?",
        "answer": "Adversarial robustness is an AI system's ability to maintain correct performance when facing intentionally crafted malicious inputs designed to fool it. Examples include slightly modified images misclassified by vision systems or text inputs that cause inappropriate outputs. It matters because: 1) Security—attackers could exploit vulnerabilities, 2) Safety—adversarial failures in autonomous vehicles or medical diagnosis could cause harm, 3) Trust—users need confidence systems work reliably. Improving robustness involves adversarial training, input validation, certified defenses, and acknowledging inherent limitations. Ethical development requires transparency about vulnerabilities and deployment in contexts considering security risks.",
        "source": "AI safety literature",
        "category": "concept"
    },

    # Societal impact
    {
        "question": "How do ethical frameworks address AI's impact on employment?",
        "answer": "Frameworks recognize AI's complex employment effects: EU AI Act classifies employment decision systems as high-risk requiring safeguards; OECD emphasizes inclusive growth and worker welfare; Partnership on AI dedicates focus to labor implications; UNESCO calls for protecting workers' rights. Common themes include: ensuring transparency in hiring/firing algorithms, preventing discrimination, maintaining human oversight for consequential employment decisions, supporting workforce transitions through education, engaging workers and unions in AI deployment decisions, and considering broader economic equity. Ethical approaches balance efficiency gains with worker dignity and social stability.",
        "source": "Multiple frameworks",
        "category": "societal"
    },
    {
        "question": "What ethical considerations arise with AI in healthcare?",
        "answer": "AI in healthcare raises critical ethical issues: 1) Patient safety—ensuring reliable, robust systems given life-or-death stakes, 2) Bias and equity—preventing algorithms from perpetuating health disparities, 3) Privacy—protecting sensitive medical data, 4) Transparency—enabling clinical interpretability and trust, 5) Autonomy—maintaining doctor-patient relationship and informed consent, 6) Liability—clarifying responsibility when AI is involved in diagnoses or treatment. Frameworks classify medical AI as high-risk requiring stringent validation, documentation, monitoring, and human oversight. Ethical deployment demands clinical trials, diverse training data, explainable models, and regulatory approval.",
        "source": "Multiple frameworks + bioethics",
        "category": "societal"
    },
    {
        "question": "What are the ethical concerns with facial recognition technology?",
        "answer": "Facial recognition raises serious ethical concerns: 1) Privacy—enabling mass surveillance and tracking, 2) Consent—often deployed without individual awareness or agreement, 3) Bias—higher error rates for women and people of color leading to discrimination, 4) Misuse—potential for authoritarian control and human rights violations, 5) Chilling effects—impacting freedom of expression and assembly. EU AI Act bans real-time biometric identification in public spaces (with narrow exceptions); UNESCO emphasizes human rights protections; multiple frameworks call for strong safeguards, transparency, and considering outright bans in certain contexts. Ethical deployment demands clear legal frameworks, accuracy across demographics, transparency, and meaningful oversight.",
        "source": "Multiple frameworks",
        "category": "societal"
    },

    # Environmental and sustainability
    {
        "question": "What is the carbon footprint of large AI models and why does it matter ethically?",
        "answer": "Training large AI models can emit substantial CO2—equivalent to hundreds of transatlantic flights. This matters ethically because: 1) Climate justice—impacts fall disproportionately on vulnerable populations, 2) Sustainability—conflicts with climate goals, 3) Resource equity—computing resources concentrated in wealthy institutions, 4) Opportunity costs—energy could serve other social needs. UNESCO uniquely emphasizes environmental ethics in AI. Responsible practices include: reporting energy consumption, optimizing model efficiency, using renewable energy, considering necessity of model size, research into efficient architectures, and carbon offsetting. The field is moving toward 'Green AI'—prioritizing efficiency alongside accuracy.",
        "source": "UNESCO + environmental AI ethics",
        "category": "environmental"
    },

    # Practical implementation
    {
        "question": "What is a Model Card and why is it important for AI ethics?",
        "answer": "Model Cards are documentation providing transparent information about machine learning models, including: intended use cases, training data characteristics, performance metrics across demographic groups, limitations and biases, ethical considerations, and appropriate deployment contexts. They're important because they: enable informed decisions about model appropriateness, facilitate accountability, reveal potential biases, support reproducibility, and help prevent misuse. Pioneered by researchers at Google, they're now recommended by frameworks emphasizing transparency. Effective Model Cards are living documents, updated as understanding evolves, accessible to both technical and non-technical stakeholders, and honest about limitations.",
        "source": "AI ethics best practices",
        "category": "implementation"
    },
    {
        "question": "How can organizations implement ethical AI governance?",
        "answer": "Implementing ethical AI governance requires: 1) Leadership commitment and clear policies aligned with frameworks, 2) Cross-functional ethics committees including diverse perspectives, 3) Ethical review processes for AI projects (like IRBs in research), 4) Training for developers on ethical principles and tools, 5) Technical infrastructure for testing bias, monitoring performance, and ensuring transparency, 6) Stakeholder engagement including affected communities, 7) Documentation standards (Model Cards, Datasheets), 8) Incident response procedures, 9) Regular audits and assessments, and 10) Accountability mechanisms with clear responsibilities. Effective governance balances innovation with risk management, adapting as AI capabilities and societal understanding evolve.",
        "source": "Multiple frameworks + governance literature",
        "category": "implementation"
    }
]

def get_by_category(category):
    """Get all guidelines by category"""
    return [item for item in ETHICAL_GUIDELINES_DATASET if item.get("category") == category]

def get_by_source(source):
    """Get all guidelines by source"""
    return [item for item in ETHICAL_GUIDELINES_DATASET if source.lower() in item.get("source", "").lower()]

def export_to_jsonl(output_file="enhanced_ai_ethics_dataset.jsonl"):
    """Export to JSONL format for training"""
    import json
    with open(output_file, 'w') as f:
        for item in ETHICAL_GUIDELINES_DATASET:
            entry = {
                "instruction": item["question"],
                "response": item["answer"],
                "metadata": {
                    "source": item.get("source", ""),
                    "category": item.get("category", "")
                }
            }
            f.write(json.dumps(entry) + '\n')
    print(f"Exported {len(ETHICAL_GUIDELINES_DATASET)} examples to {output_file}")

if __name__ == "__main__":
    export_to_jsonl()
    print(f"\nDataset statistics:")
    print(f"Total examples: {len(ETHICAL_GUIDELINES_DATASET)}")
    print(f"Categories: {set(item.get('category') for item in ETHICAL_GUIDELINES_DATASET)}")
    print(f"Sources covered: EU AI Act, OECD, Partnership on AI, UNESCO")
