import re
from datetime import datetime, timedelta
from typing import Tuple, Optional

# Reference date for the corpus
REFERENCE_DATE = datetime(2026, 7, 11)

def parse_temporal_cues(query: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Deterministically parses temporal cues from a query string relative to a fixed REFERENCE_DATE.
    Returns (start_date, end_date) if successful, otherwise (None, None).
    """
    query_lower = query.lower()
    
    # Valid corpus months mapping
    months = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12
    }
    
    # 1. Relative phrases
    if re.search(r'\b(?:last month|pichle mahine|pichhla mahina)\b', query_lower):
        # Last month relative to July 2026 is June 2026
        return datetime(2026, 6, 1), datetime(2026, 6, 30, 23, 59, 59)
    if re.search(r'\b(?:this month|iss mahine|is mahine)\b', query_lower):
        # This month is July 2026
        return datetime(2026, 7, 1), datetime(2026, 7, 31, 23, 59, 59)
    if re.search(r'\b(?:last week|pichle hafte|pichhla hafta)\b', query_lower):
        # Reference is July 11 (Saturday). Last week would be 7-14 days ago.
        start = (REFERENCE_DATE - timedelta(days=14)).replace(hour=0, minute=0, second=0)
        end = (REFERENCE_DATE - timedelta(days=7)).replace(hour=23, minute=59, second=59)
        return start, end
    if re.search(r'\b(?:this week|iss hafte|is hafte)\b', query_lower):
        start = (REFERENCE_DATE - timedelta(days=6)).replace(hour=0, minute=0, second=0)
        end = REFERENCE_DATE.replace(hour=23, minute=59, second=59)
        return start, end
    if re.search(r'\b(?:yesterday|kal)\b', query_lower):
        start = (REFERENCE_DATE - timedelta(days=1)).replace(hour=0, minute=0, second=0)
        end = start.replace(hour=23, minute=59, second=59)
        return start, end
    if re.search(r'\btomorrow\b', query_lower):
        start = (REFERENCE_DATE + timedelta(days=1)).replace(hour=0, minute=0, second=0)
        end = start.replace(hour=23, minute=59, second=59)
        return start, end

    # 2. Explicit Date (e.g. 12 June or June 12)
    date_pattern_1 = r'\b(\d{1,2})\s*(?:st|nd|rd|th)?\s+(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\b'
    match = re.search(date_pattern_1, query_lower)
    day_str, month_str = None, None
    
    if match:
        day_str, month_str = match.groups()
    else:
        date_pattern_2 = r'\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\s+(\d{1,2})\b'
        match = re.search(date_pattern_2, query_lower)
        if match:
            month_str, day_str = match.groups()

    if day_str and month_str:
        day = int(day_str)
        month = months[month_str]
        try:
            target_date = datetime(2026, month, day)
            return target_date, target_date.replace(hour=23, minute=59, second=59)
        except ValueError:
            pass # Invalid date fallback

    # 3. Explicit month ONLY (e.g. "in March")
    for m_name, m_num in months.items():
        if re.search(rf'\b(?:in|on|during|for)\s+{m_name}\b', query_lower) or re.search(rf'\b{m_name}\b', query_lower):
            # Calculate last day of the month
            if m_num in [1, 3, 5, 7, 8, 10, 12]:
                last_day = 31
            elif m_num == 2:
                last_day = 28 # 2026 is not a leap year
            else:
                last_day = 30
            
            return datetime(2026, m_num, 1), datetime(2026, m_num, last_day, 23, 59, 59)

    # 4. Weekdays (e.g. "on Saturday", "last Monday")
    # IMPORTANT: Only match when the weekday is preceded by an explicit temporal modifier
    # (on, last, this, next, past). A bare weekday used as a descriptor (e.g. "the Saturday plan",
    # "Saturday meeting") is semantic content, NOT a date filter constraint.
    weekdays = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
    for wd_name, wd_num in weekdays.items():
        if re.search(rf'\b(?:on|last|this|next|past)\s+{wd_name}\b', query_lower):
            days_ago = (REFERENCE_DATE.weekday() - wd_num) % 7
            target_date = REFERENCE_DATE - timedelta(days=days_ago)
            start = target_date.replace(hour=0, minute=0, second=0)
            end = target_date.replace(hour=23, minute=59, second=59)
            return start, end

    return None, None
