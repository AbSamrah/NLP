from owlready2 import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import graphviz

def define_ontology(iri="http://example.org/hotel_reviews.owl", filename="hotel_ontology.owl"):
    """Defines the classes and properties of the ontology."""
    onto = get_ontology(iri)

    with onto:
        class HotelAspect(Thing): pass
        class Sentiment(Thing): pass
        class Review(Thing): pass
        
        class Room(HotelAspect): pass
        class Staff(HotelAspect): pass
        class Food(HotelAspect): pass
        class Service(HotelAspect): pass
        class Location(HotelAspect): pass
        
        class Positive(Sentiment): pass
        class Negative(Sentiment): pass
        class Neutral(Sentiment): pass

        class mentions_aspect(ObjectProperty):
            domain = [Review]; range = [HotelAspect]
        class has_sentiment(ObjectProperty):
            domain = [Review]; range = [Sentiment]
        class has_text(DataProperty):
            domain = [Review]; range = [str]

    onto.save(file=filename, format="rdfxml")
    return onto

def populate_ontology(df, onto_path="hotel_ontology.owl", save_path="hotel_ontology_populated.owl"):
    """Populates the ontology using the DataFrame and VADER sentiment."""
    analyzer = SentimentIntensityAnalyzer()
    onto = get_ontology(onto_path).load()
    
    ASPECT_KEYWORDS = {
        "Room": ["room", "bed", "bathroom", "shower", "ac"],
        "Staff": ["staff", "manager", "reception", "desk"],
        "Food": ["food", "breakfast", "dinner", "meal"],
        "Service": ["service", "check-in", "booking"],
        "Location": ["location", "view", "area", "walk"]
    }

    with onto:
        for index, row in df.iterrows():
            review_id = f"Review_{row.get('review_id', index)}"
            new_review = onto.Review(review_id)
            new_review.has_text.append(str(row['text']))
            
            # Sentiment Rule
            score = analyzer.polarity_scores(str(row['text']))['compound']
            if score >= 0.05:
                new_review.has_sentiment.append(onto.Positive("OverallPositive"))
            elif score <= -0.05:
                new_review.has_sentiment.append(onto.Negative("OverallNegative"))
            else:
                new_review.has_sentiment.append(onto.Neutral("OverallNeutral"))
            
            # Aspect Rule
            txt = str(row['text']).lower()
            for aspect, keywords in ASPECT_KEYWORDS.items():
                if any(k in txt for k in keywords):
                    AspectClass = getattr(onto, aspect)
                    new_review.mentions_aspect.append(AspectClass(f"{aspect}_in_{review_id}"))

    onto.save(file=save_path, format="rdfxml")
    print("Ontology populated and saved.")
    return onto

def visualize_ontology_structure(onto_path="hotel_ontology.owl"):
    """Generates a graphviz visualization of the ontology structure."""
    try:
        onto = get_ontology(onto_path).load()
    except:
        print("Ontology file not found.")
        return

    dot = graphviz.Digraph(comment='Hotel Review Ontology')
    dot.attr(rankdir='BT') 

    for cls in onto.classes():
        if cls is not Thing:
            dot.node(cls.name, cls.name, shape='ellipse', style='filled', color='lightblue')
            for super_cls in cls.is_a:
                if isinstance(super_cls, ThingClass) and super_cls is not Thing:
                    dot.edge(cls.name, super_cls.name, label="is_a")

    for prop in onto.object_properties():
        for d in prop.domain:
            for r in prop.range:
                if d is not Thing and r is not Thing:
                    dot.edge(d.name, r.name, label=prop.name, color='red', style='dashed')
    
    dot.render('hotel_ontology_structure', format='png', view=False)
    print("Graph saved as hotel_ontology_structure.png")
    return dot