#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenLIF_Exponential_triple.h"

#include "../source_file_realtime_v1_async/connections/inc/Interconnections.h"

TimeDrivenLIF_Exponential_triple::TimeDrivenLIF_Exponential_triple()
    : TimeDrivenLIF_Exponential_double() {
    this->setModelName(std::string("TimeDrivenLIF_Exponential_triple"));
}

TimeDrivenLIF_Exponential_triple::TimeDrivenLIF_Exponential_triple(int timesteps)
    : TimeDrivenLIF_Exponential_double(timesteps) {
    this->setModelName(std::string("TimeDrivenLIF_Exponential_triple"));
}

void TimeDrivenLIF_Exponential_triple::CheckType(Interconnections* inter) {
    if (inter->type == 2) {
        // Legacy CPU keeps the public triple model creatable. NMDA is accepted
        // as a conductance input here and shares the excitatory accumulation
        // path until the old CPU integrator grows a separate NMDA state.
        this->Excited = true;
        return;
    }
    TimeDrivenLIF_Exponential_double::CheckType(inter);
}

void TimeDrivenLIF_Exponential_triple::SetParameters(std::map<std::string, boost::any> parametermap,
                                                     float basetimestep) {
    std::map<std::string, boost::any>::iterator iter = parametermap.find("E_ampa");
    if (iter != parametermap.end()) {
        this->E_ampa = boost::any_cast<float>(iter->second);
        this->Eexc = this->E_ampa;
        parametermap.erase(iter);
    }
    iter = parametermap.find("ampa_tau");
    if (iter != parametermap.end()) {
        this->ampa_tau = boost::any_cast<float>(iter->second);
        this->gexc_tau = this->ampa_tau;
        parametermap.erase(iter);
    }
    iter = parametermap.find("E_gaba");
    if (iter != parametermap.end()) {
        this->E_gaba = boost::any_cast<float>(iter->second);
        this->Einhibitory = this->E_gaba;
        parametermap.erase(iter);
    }
    iter = parametermap.find("gaba_tau");
    if (iter != parametermap.end()) {
        this->gaba_tau = boost::any_cast<float>(iter->second);
        this->ginh_tau = this->gaba_tau;
        parametermap.erase(iter);
    }
    iter = parametermap.find("nmda_tau");
    if (iter != parametermap.end()) {
        this->nmda_tau = boost::any_cast<float>(iter->second);
        parametermap.erase(iter);
    }
    TimeDrivenLIF_Exponential_double::SetParameters(parametermap, basetimestep);
}

std::map<std::string, boost::any> TimeDrivenLIF_Exponential_triple::getParameters() {
    std::map<std::string, boost::any> parametermap =
        TimeDrivenLIF_Exponential_double::getParameters();
    parametermap["E_ampa"] = this->E_ampa;
    parametermap["ampa_tau"] = this->ampa_tau;
    parametermap["E_gaba"] = this->E_gaba;
    parametermap["gaba_tau"] = this->gaba_tau;
    parametermap["nmda_tau"] = this->nmda_tau;
    return parametermap;
}
