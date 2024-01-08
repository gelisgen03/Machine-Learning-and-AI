total_cost=0#toplam ev maliyeti
portion_down_payment=0.25#ev pesinati
current_savings=0#toplam birikim
annual_salary=0#aylik maas
portion_saved=0# pesinat inin biriktirilecek maas yuzdesi
r=0.04#Yatirimlarin yillik getirisi
mount=0
semi_annual_raise=0

annual_salary=float(input("Yillik Maasinizi giriniz"))
portion_saved=float(input("Maasinizin yuzde kacini birikim yapacaksiniz"))
total_cost=float(input("Hayalinizdeki evin maliyeti ne kadar"))
semi_annual_raise=float(input("Yari yillik maas artisiniz"))

salary=annual_salary/12

while(1):
    
    if mount%6==1 and mount!=1 :
        salary=salary+salary*semi_annual_raise

    mountly_saved=salary*portion_saved
    current_savings=mountly_saved+current_savings*r/12+current_savings
    mount+=1
    if current_savings>=total_cost*portion_down_payment:
        print("Ay Sayisi: ",mount)
        break
    





