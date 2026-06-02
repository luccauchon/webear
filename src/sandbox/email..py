import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def envoyer_courriel(
        expediteur,
        destinataire,
        sujet,
        corps,
        mot_de_passe_app,
        serveur_smtp="smtp.gmail.com",
        port_smtp=587
):
    # Construire le message
    msg = MIMEMultipart()
    msg['From'] = expediteur
    msg['To'] = destinataire
    msg['Subject'] = sujet
    msg.attach(MIMEText(corps, 'plain'))

    # Serveur SMTP
    server = smtplib.SMTP(serveur_smtp, port_smtp)
    server.starttls()
    server.login(expediteur, mot_de_passe_app)

    # Envoyer le courriel
    texte = msg.as_string()
    server.sendmail(expediteur, destinataire, texte)
    server.quit()


# Utilisation
expediteur = "luccauchon@gmail.com"
destinataire = "luccauchon@gmail.com"
sujet = "Test envoi de courriel"
corps = "Bonjour, ceci est un test."
mot_de_passe_app = "thhy qvae fbsb zsbe"

envoyer_courriel(expediteur, destinataire, sujet, corps, mot_de_passe_app)