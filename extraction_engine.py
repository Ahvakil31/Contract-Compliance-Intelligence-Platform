# extraction_engine.py
import re
from typing import Dict, List, Optional
import spacy
from config import SPACY_MODEL

class ContractExtractor:
    """Enhanced contract extraction with multiple strategies."""
    
    def __init__(self):
        try:
            self.nlp = spacy.load(SPACY_MODEL)
            print(f"✅ spaCy model '{SPACY_MODEL}' loaded successfully")
        except Exception as e:
            print(f"⚠️ spaCy model '{SPACY_MODEL}' not found. NER features disabled. Error: {e}")
            self.nlp = None
    
    def extract_client_name(self, text: str) -> str:
        """Extract client/party names using multiple strategies."""
        patterns = [
            r"(?:LICENSOR|LESSOR|PARTY OF FIRST PART|MR\.|MRS\.|MS\.|SHRI|SMT\.)\s*[:\s]*([A-Za-z\s\.]{3,60}?)(?:\n|,|\.|aged|hereinafter)",
            r"BETWEEN\s+(?:[0-9\.\s]+)?\s*([A-Za-z\s\.]{5,60}?)(?:\s*\(|\n|,|and)",
            r"(?:MR\.|MRS\.|MS\.|DR\.|PROF\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"by\s+and\s+between\s+([A-Za-z\s\.]{5,80}?)\s+(?:and|\(|\n)",
            r"(?:first\s+party|party\s+of\s+first\s+part)\s*(?:is|:)?\s*([A-Za-z\s\.]{3,60}?)",
            r"Licensor\s*[:]?\s*([A-Za-z\s\.]{3,60}?)(?:\n|,|\.|aged)",
            r"([A-Z][A-Z\s\.]{3,40}?)(?:\s+\(hereinafter\s+referred\s+to\s+as)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                invalid = ["shall deposit a sum of rs", "counterparty", "this", "agreement", "party", "the"]
                if len(name) > 3 and not any(w in name.lower() for w in invalid):
                    return name
        
        # Try spaCy NER as fallback
        if self.nlp:
            doc = self.nlp(text[:5000])
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON" and len(ent.text) > 3]
            if persons:
                return persons[0]
        
        return "Primary Contracting Party"
    
    def extract_counterparty_name(self, text: str) -> str:
        """Extract counterparty name with improved patterns."""
        patterns = [
            r"(?:LICENSEE|LESSEE)\s*[:]\s*([A-Za-z\s\.]{3,60}?)(?:\n|,|\.|aged|hereinafter)",
            r"(?:LICENSEE|LESSEE)\s+(?:is|:)?\s*([A-Za-z\s\.]{3,60}?)(?:\n|,|\.)",
            r"AND\s+([A-Za-z\s\.]{5,60}?)(?:\s*\(|\n|,|as|hereinafter)",
            r"(?:PARTY\s+OF\s+SECOND\s+PART)\s*(?:is|:)?\s*([A-Za-z\s\.]{3,60}?)",
            r"Licensee\s*[:]?\s*([A-Za-z\s\.]{3,60}?)(?:\n|,|\.|aged)",
            r"between\s+[A-Za-z\s\.]+\s+and\s+([A-Za-z\s\.]{5,60}?)(?:\s*\(|\n|,|$)",
            r"(?:second\s+party|party\s+of\s+second\s+part)\s*(?:is|:)?\s*([A-Za-z\s\.]{3,60}?)",
            r"([A-Z][A-Z\s\.]{3,40}?)\s+(?:\(hereinafter\s+referred\s+to\s+as\s+the\s+LICENSEE)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                invalid = ["shall deposit a sum of rs", "counterparty", "the", "this", "agreement", "and"]
                if len(name) > 3 and not any(w in name.lower() for w in invalid):
                    return name
        
        if self.nlp:
            doc = self.nlp(text[:5000])
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON" and len(ent.text) > 3]
            if len(persons) > 1:
                return persons[1]
            elif persons:
                return f"Counterparty: {persons[0]}"
        
        # Check for "and" with a name after it
        and_match = re.search(r"and\s+([A-Za-z\s\.]{5,50}?)(?:\s*\(|\n|,|$)", text, re.IGNORECASE)
        if and_match:
            name = and_match.group(1).strip()
            if len(name) > 3:
                return name
        
        return "Counterparty Entity"
    
    def extract_effective_date(self, text: str) -> str:
        """Extract effective date with improved patterns."""
        patterns = [
            r"(?:effective\s+date|dated|as\s+of|executed\s+on)\s*[:]?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})\s*(?:this|agreement)",
            r"this\s+agreement\s+is\s+made\s+on\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"made\s+and\s+entered\s+into\s+on\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"executed\s+at\s+[A-Za-z]+\s+on\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"this\s+(\d{1,2})\s+day\s+of\s+([A-Za-z]+),\s+(\d{4})",
            r"dated\s+this\s+(\d{1,2})\s+day\s+of\s+([A-Za-z]+)\s+(\d{4})",
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
            r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})",
            r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:
                    day, month, year = match.groups()
                    return f"{day} {month} {year}"
                return match.group(1).strip()
        
        if self.nlp:
            doc = self.nlp(text[:5000])
            dates = [ent.text for ent in doc.ents if ent.label_ == "DATE" and len(ent.text) > 4]
            if dates:
                return dates[0]
        
        return "Execution Date"
    
    def extract_contract_type(self, text: str) -> str:
        """Detect contract type with weighted scoring."""
        text_lower = text.lower()
        types = {
            "Leave and License Agreement": ["leave and license", "licensor", "licensee", "leave & license", "leave and licence", "leave & licence"],
            "Rental Agreement": ["rental", "tenancy", "rent", "landlord", "tenant"],
            "Non-Disclosure Agreement": ["non-disclosure", "confidentiality", "nda", "confidential"],
            "Services Agreement": ["service", "msa", "statement of work", "sow"],
            "Employment Agreement": ["employment", "employee", "employer", "compensation", "salary"],
            "Partnership Agreement": ["partnership", "joint venture", "partner"],
            "Consultancy Agreement": ["consultant", "consultancy", "professional services"],
            "Affiliate Agreement": ["affiliate", "chase", "bank", "partnership"],
            "Sale Deed": ["sale deed", "purchase", "vendor", "purchaser", "seller"],
            "Commercial Agreement": ["commercial", "business", "trade"]
        }
        
        scores = {}
        for contract_type, keywords in types.items():
            score = sum(2 for kw in keywords if kw in text_lower)
            scores[contract_type] = score
        
        if scores:
            best_match = max(scores, key=scores.get)
            if scores[best_match] > 0:
                return best_match
        return "Commercial Agreement"
    
    def extract_term_duration(self, text: str) -> str:
        """Extract contract term duration."""
        patterns = [
            r"term\s+(?:of\s+)?(?:this\s+)?agreement\s+(?:is|shall\s+be)\s+(\d+)\s+(?:months?|years?)",
            r"(\d+)\s*(?:months?|years?)\s+term",
            r"initial\s+term\s+of\s+(\d+)\s+(?:months?|years?)",
            r"from\s+(\d+\s+(?:months?|years?))",
            r"for\s+a\s+period\s+of\s+(\d+)\s+(?:months?|years?)",
            r"valid\s+for\s+(\d+)\s+(?:months?|years?)",
            r"duration\s+(?:of|for)\s+(\d+)\s+(?:months?|years?)",
            r"period\s+of\s+(\d+)\s+months",
            r"(\d+)\s+month",
            r"(\d+)\s+year"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "Specified in Agreement"
    
    def extract_notice_period(self, text: str) -> str:
        """Extract notice period for termination."""
        patterns = [
            r"(\d+)\s*days?\s+(?:notice|prior)",
            r"notice\s+period\s+of\s+(\d+)\s*days?",
            r"terminate\s+with\s+(\d+)\s*days?\s+notice",
            r"provid(?:e|ing)\s+(\d+)\s*days?\s+notice",
            r"(\d+)\s*days?\s+written\s+notice",
            r"notice\s+of\s+(\d+)\s+days",
            r"(\d+)\s+days(?:'?)\s+notice"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "30"
    
    def extract_governing_law(self, text: str) -> str:
        """Extract governing law."""
        patterns = [
            r"(?:governed\s+by|construed\s+in\s+accordance\s+with)\s+(?:the\s+laws\s+of\s+)?([A-Za-z\s]{3,40}?)(?:\.|;|,|\n|and)",
            r"(?:subject\s+to\s+the\s+jurisdiction\s+of)\s+([A-Za-z\s]{3,40}?)(?:\.|;|,|\n)",
            r"(Maharashtra\s*Rent\s*Control\s*Act[A-Za-z0-9\s]*)",
            r"(Indian\s*Contract\s*Act[A-Za-z0-9\s]*)",
            r"laws\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(State\s+of\s+[A-Za-z]+)"
        ]
        
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                res = m.group(1).strip()
                if len(res) > 3 and not any(w in res.lower() for w in ["this", "agreement", "party"]):
                    return res
        
        text_lower = text.lower()
        if "maharashtra" in text_lower:
            return "Maharashtra Rent Control Act 1999"
        elif "delhi" in text_lower:
            return "Delhi Rent Control Act"
        elif "california" in text_lower:
            return "California Civil Code"
        elif "new york" in text_lower:
            return "New York State Laws"
        elif "delaware" in text_lower:
            return "State of Delaware"
        elif "england" in text_lower or "uk" in text_lower:
            return "English Law"
        
        return "Governing Jurisdiction Clause (Refer to Execution Terms)"
    
    def extract_jurisdiction(self, text: str) -> str:
        """Extract jurisdiction/venue."""
        patterns = [
            r"courts\s+(?:at|in)\s+([A-Za-z\s]{3,30}?)(?:\s+shall\s+have|\.|;|,|\n)",
            r"exclusive\s+jurisdiction\s+of\s+(?:the\s+courts\s+at|in\s+)?([A-Za-z\s]{3,30}?)(?:\.|;|,|\n)",
            r"venue\s+shall\s+be\s+([A-Za-z\s]{3,30}?)(?:\.|;|,|\n)",
            r"jurisdiction\s*[:]?\s*([A-Za-z\s]{3,40}?)(?:\.|;|,|\n)"
        ]
        
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                res = m.group(1).strip()
                if len(res) > 2:
                    return f"Competent Courts at {res.title()}"
        
        if self.nlp:
            doc = self.nlp(text[:8000])
            gpes = [ent.text.strip() for ent in doc.ents if ent.label_ == "GPE" and len(ent.text.strip()) > 2]
            if gpes:
                return f"Competent Courts at {gpes[0].title()}"
        
        # Check for city names
        cities = ["mumbai", "delhi", "bangalore", "chennai", "hyderabad", "pune", "kolkata", "new york"]
        for city in cities:
            if city in text.lower():
                return f"Competent Courts at {city.title()}"
        
        return "Designated Local Competent Courts"
    
    def extract_adr(self, text: str) -> str:
        """Extract ADR/Arbitration provisions."""
        text_lower = text.lower()
        
        if "arbitration" in text_lower:
            arb_match = re.search(r"arbitration\s+in\s+accordance\s+with\s+([A-Za-z0-9\s]{5,50}?)(?:\.|;|\n)", text, re.IGNORECASE)
            if arb_match:
                return f"Arbitration under {arb_match.group(1).strip()}"
            
            arb_match2 = re.search(r"(?:disputes?\s+shall\s+be\s+)?referred\s+to\s+(?:a\s+)?([A-Za-z\s]{3,30}?)(?:\s+arbitration|\.|;|\n)", text, re.IGNORECASE)
            if arb_match2:
                return f"Arbitration by {arb_match2.group(1).strip()}"
            
            return "Sole Arbitrator / Arbitration Act Mechanics"
        elif "competent authority" in text_lower:
            return "Competent Authority under Rent Control Act"
        elif "mediation" in text_lower:
            return "Mutual Conciliation & Mediation"
        elif "lok adalat" in text_lower:
            return "Lok Adalat / Permanent Lok Adalat"
        
        return "Direct Executive Negotiation / Court Dispute Settlement"
    
    def extract_deposit(self, text: str) -> str:
        """Extract security deposit amount with improved patterns."""
        patterns = [
            r"(?:security\s+deposit|deposit)\s*(?:of|amount)?\s*[:]?\s*(?:Rs\.?\s*)?([\d,]+)",
            r"deposit\s+amount\s*(?:Rs\.?\s*)?([\d,]+)",
            r"pay\s*(?:a|an)\s+(?:security\s+)?deposit\s+of\s*(?:Rs\.?\s*)?([\d,]+)",
            r"deposit\s+of\s+([\d,]+)",
            r"Rs\.\s*([\d,]+)\s*(?:as|towards)\s+deposit",
            r"interest\s+free\s+deposit\s+of\s*(?:Rs\.?\s*)?([\d,]+)",
            r"deposit\s+amounting\s+to\s+(?:Rs\.?\s*)?([\d,]+)",
            r"deposit\s+sum\s+of\s*(?:Rs\.?\s*)?([\d,]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = match.group(1).strip()
                return f"Rs. {amount}"
        
        # Look for "deposit" with amount nearby
        deposit_pattern = r"Rs\.\s*([\d,]+)"
        if "deposit" in text.lower():
            amounts = re.findall(deposit_pattern, text, re.IGNORECASE)
            if amounts:
                for amount in amounts:
                    pos = text.find(amount)
                    context = text[max(0, pos-150):min(len(text), pos+150)]
                    if "deposit" in context.lower():
                        return f"Rs. {amount}"
        
        return "Agreed Sum"
    
    def extract_rent(self, text: str) -> str:
        """Extract rent amount with improved patterns."""
        patterns = [
            r"(?:monthly\s+consideration|license\s+fee|rent)\s*(?:of|amount)?\s*[:]?\s*(?:Rs\.?\s*)?([\d,]+)",
            r"Rs\.\s*([\d,]+)\s*/?\s*per\s*month",
            r"monthly\s+rent\s+of\s*(?:Rs\.?\s*)?([\d,]+)",
            r"consideration\s+of\s+Rs\.\s*([\d,]+)",
            r"license\s+fee\s+of\s*(?:Rs\.?\s*)?([\d,]+)",
            r"shall\s+pay\s+(?:a|an)\s+(?:monthly\s+)?(?:rent|license\s+fee)\s+of\s*(?:Rs\.?\s*)?([\d,]+)",
            r"rental\s+amount\s+Rs\.\s*([\d,]+)",
            r"payable\s+@\s+Rs\.\s*([\d,]+)",
            r"at\s+the\s+rate\s+of\s+Rs\.\s*([\d,]+)",
            r"Rs\.\s*([\d,]+)\s+per\s+month",
            r"monthly\s+payable\s+amount\s*[:]?\s*(?:Rs\.?\s*)?([\d,]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = match.group(1).strip()
                return f"Rs. {amount} / Month"
        
        # Look for amounts with "per month" nearby
        amount_pattern = r"Rs\.\s*([\d,]+)"
        amounts = re.findall(amount_pattern, text, re.IGNORECASE)
        if amounts:
            for amount in amounts:
                context_start = max(0, text.find(amount) - 100)
                context_end = min(len(text), text.find(amount) + 100)
                context = text[context_start:context_end]
                if "month" in context.lower():
                    return f"Rs. {amount} / Month"
        
        return "Standard Consideration"
    
    def extract_execution_status(self, text: str) -> str:
        """Extract execution status."""
        text_lower = text.lower()
        if any(k in text_lower for k in ["in witness whereof", "signed", "executed", "signature"]):
            if any(k in text_lower for k in ["executed", "fully executed"]):
                return "Fully Executed"
            return "Signed"
        return "Draft Agreement"
    
    def extract_all(self, text: str) -> Dict[str, any]:
        """Extract all contract details at once."""
        clean_text = self._clean_text(text)
        
        return {
            "client_name": self.extract_client_name(clean_text),
            "counterparty_name": self.extract_counterparty_name(clean_text),
            "effective_date": self.extract_effective_date(clean_text),
            "execution_status": self.extract_execution_status(clean_text),
            "contract_type": self.extract_contract_type(clean_text),
            "term_duration": self.extract_term_duration(clean_text),
            "notice_period": self.extract_notice_period(clean_text),
            "governing_law": self.extract_governing_law(clean_text),
            "jurisdiction": self.extract_jurisdiction(clean_text),
            "adr": self.extract_adr(clean_text),
            "deposit": self.extract_deposit(clean_text),
            "rent": self.extract_rent(clean_text)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text for extraction."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'%PDF-\d\.\d.*?(?=\n|\r)', '', text)
        text = re.sub(r'<<.*?>>', '', text, flags=re.DOTALL)
        text = re.sub(r'[^\x20-\x7E\n\r]', ' ', text)
        return text.strip()