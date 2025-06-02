import os
import glob
from datetime import datetime
import win32com.client

# Set the directory where the PDFs are stored
pdf_dir = os.path.dirname(os.path.abspath(__file__))

# Find the latest backtest and battery optimization PDF files
def find_latest_pdf(pattern):
    files = glob.glob(os.path.join(pdf_dir, pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

latest_backtest = find_latest_pdf('backtest_*.pdf')
latest_optimization = find_latest_pdf('battery_optimization_*.pdf')

# Create Outlook email
def create_outlook_email():
    outlook = win32com.client.Dispatch('Outlook.Application')
    mail = outlook.CreateItem(0)
    mail.Subject = 'Wekelijkse Imbalans Sturing Backtest'
    # Set only HTMLBody to avoid duplicate text and keep signature
    body_html = (
        'Hey JF, Tom en Briac,<br><br>'
        'In bijlage vinden jullie de wekelijkse backtest resultaten voor de imbalans sturing.<br>'
        'De volgende documenten zijn toegevoegd:<br>'
        '- Het backtest rapport van de afgelopen week<br>'
        '- Het optimalisatie rapport van de batterij voor de afgelopen week<br><br>'
        'Laat gerust weten als er vragen of opmerkingen zijn.<br><br>'
    )
    mail.HTMLBody = body_html + mail.HTMLBody  # Prepend to keep signature
    # Do not set mail.Body, only HTMLBody
    mail.To = 'jean-francois.williame@eneco.com; tom.strosse@eneco.com; briac.puel@eneco.com'
    if latest_backtest:
        mail.Attachments.Add(latest_backtest)
    if latest_optimization:
        mail.Attachments.Add(latest_optimization)
    mail.Display()  # Opens the email for review

if __name__ == '__main__':
    create_outlook_email()
