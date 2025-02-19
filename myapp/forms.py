# myapp/forms.py
from django import forms
from .models import Dataset

class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['file']

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            extension = file.name.split('.')[-1].lower()
            if extension not in ['csv', 'xls', 'xlsx']:
                raise forms.ValidationError("Only CSV and Excel files are allowed.")
        return file

class KSelectionForm(forms.Form):
    K_CHOICES = [(i, str(i)) for i in range(2, 8)]
    k_value = forms.ChoiceField(choices=K_CHOICES, label='Number of Clusters', initial=4)
    
    
# forms.py
from django import forms

class DateRangeForm(forms.Form):
    start_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    end_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))

class ClusterSelectionForm(forms.Form):
    num_clusters = forms.IntegerField(min_value=2, max_value=7, initial=4)
